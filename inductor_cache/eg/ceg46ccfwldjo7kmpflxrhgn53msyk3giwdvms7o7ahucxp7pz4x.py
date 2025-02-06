# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/qm/cqmugohlsno2dnwonorw5rljew5xqvljdmnfq6m2if7ikzpt6oqp.py
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
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/sg/csg7v36yyd53yj4ysvxje455pptjyatplxkc3hr3wuyt5nnwvhqz.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => add_2, clamp_max, clamp_min, div, mul_3
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %clamp_max), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/ue/cueloq6etzlrkmanzmdl72rf5xzqnmdgnhl5l4ux4fyg3uptpj44.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_4, mul_5, mul_6, sub_1
#   input_6 => relu
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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


# kernel path: inductor_cache/n4/cn4kwrsnax3lzo5zqc4l7eygxkl73lcztdqaojwbb2qihqkkj3f4.py
# Topologically Sorted Source Nodes: [scale], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   scale => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu, [-1, -2], True), kwargs = {})
triton_red_fused_mean_5 = async_compile.triton('triton_red_fused_mean_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 16)
    x1 = xindex // 16
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 2048*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pe/cped5dgou4sxlljmyptjupzb5fbfrvvblioc3xucloq55joazpzg.py
# Topologically Sorted Source Nodes: [scale], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   scale => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu, [-1, -2], True), kwargs = {})
triton_per_fused_mean_6 = async_compile.triton('triton_per_fused_mean_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_6(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 16)
    x1 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*r2 + 32*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ng/cngepiliv7lzktyybn4sgtl4pjfico5ut64onxea377mr6g4urez.py
# Topologically Sorted Source Nodes: [scale_1, scale_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   scale_1 => convolution_2
#   scale_2 => relu_1
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_12, %primals_13, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_relu_7 = async_compile.triton('triton_poi_fused_convolution_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fg/cfglasuwc4fy632hvwt6aauuupai2vjnrpasksujop2vjquhx3r4.py
# Topologically Sorted Source Nodes: [scale_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   scale_3 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_8 = async_compile.triton('triton_poi_fused_convolution_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fk/cfkcgi766ceheajb4enez6f6msqrszxv64qnj2tmrrhx5aqzgtxm.py
# Topologically Sorted Source Nodes: [scale_4, input_7], Original ATen: [aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_7 => mul_7
#   scale_4 => add_5, clamp_max_1, clamp_min_1, div_1
# Graph fragment:
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, 3), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_5, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_1, 6), kwargs = {})
#   %mul_7 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %relu), kwargs = {})
triton_poi_fused_hardsigmoid_mul_9 = async_compile.triton('triton_poi_fused_hardsigmoid_mul_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_mul_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardsigmoid_mul_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/p2/cp2wprrfkkb2ecfub3lanxvuuwihxcnotank3u3cqha7favsxbou.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_9 => add_7, mul_10, mul_9, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/ja/cja6szlaam2goso45dtbe5ysi5d2vnfr7rrsxidj7435na7w7lqy.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_11 => add_9, mul_12, mul_13, sub_3
#   input_12 => relu_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_25), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_29), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_31), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 72)
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


# kernel path: inductor_cache/ap/capn4x5uteqa6oz2xof7h7ubqqptnfraxkfja43dqkd4fov72ls7.py
# Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_14 => add_11, mul_15, mul_16, sub_4
#   input_15 => relu_3
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_33), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_37), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_39), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 72)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mo/cmoy7yj3jxywkbnk7emll6xj5sfgne6wy5q4a3vigxuirrl7kafm.py
# Topologically Sorted Source Nodes: [input_17], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_17 => add_13, mul_18, mul_19, sub_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_41), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_45), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_47), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ft/cftebe63cthw22ltnywxxdjifznj22mfmofphj2kofsjwy54ufqp.py
# Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_19 => add_15, mul_21, mul_22, sub_6
#   input_20 => relu_4
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_49), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_55), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 88)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nn/cnna2pakhtvbon5djb5wbdrfsboeq7tm5twgkbxudu5pz7tnmjmv.py
# Topologically Sorted Source Nodes: [input_25, result], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_25 => add_19, mul_27, mul_28, sub_8
#   result => add_20
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_65), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_27, %unsqueeze_69), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %unsqueeze_71), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %add_13), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s3/cs3seiyuothwe7nh5f7shpx5gpch5mkutcn7umyzeetxccnc7ekj.py
# Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   input_27 => add_22, mul_30, mul_31, sub_9
#   input_28 => add_23, clamp_max_2, clamp_min_2, div_2, mul_32
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_73), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %unsqueeze_77), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_79), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, 3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_23, 0), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %clamp_max_2), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_32, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 96)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/vf/cvf7tqu2ilum6nklqml6qkucdhvkhblft7zcluoj7pcbyt2kdysn.py
# Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_30 => add_25, mul_34, mul_35, sub_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_81), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_85), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_87), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5x/c5xof3c6ewxjv5ojnpb4pbgdt6kfk47poww6bnblk7sba255pgzk.py
# Topologically Sorted Source Nodes: [input_31, scale_5], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   input_31 => add_26, clamp_max_3, clamp_min_3, div_3, mul_36
#   scale_5 => mean_1
# Graph fragment:
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, 3), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_26, 0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, %clamp_max_3), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_36, 6), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_3, [-1, -2], True), kwargs = {})
triton_per_fused_hardswish_mean_18 = async_compile.triton('triton_per_fused_hardswish_mean_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_18(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 96)
    x1 = xindex // 96
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*r2 + 1536*x1), xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 16.0
    tmp15 = tmp13 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/25/c25k6be3ozuzghg3v2qnmj5irgd57mulw43pady7ijm4f6vodobe.py
# Topologically Sorted Source Nodes: [scale_6, scale_7], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   scale_6 => convolution_13
#   scale_7 => relu_6
# Graph fragment:
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_1, %primals_61, %primals_62, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_13,), kwargs = {})
triton_poi_fused_convolution_relu_19 = async_compile.triton('triton_poi_fused_convolution_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xu/cxu7b67ddxlr24yoqq6rggzfhqhle5ioytw2obswsa7mgpx5j3g4.py
# Topologically Sorted Source Nodes: [scale_8], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   scale_8 => convolution_14
# Graph fragment:
#   %convolution_14 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_6, %primals_63, %primals_64, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2q/c2qkkdyhvyqahl74rd3lz5xrjnnx7nedscqgsrnukyokeyn7nhzt.py
# Topologically Sorted Source Nodes: [input_31, scale_9, input_32], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_31 => add_26, clamp_max_3, clamp_min_3, div_3, mul_36
#   input_32 => mul_37
#   scale_9 => add_27, clamp_max_4, clamp_min_4, div_4
# Graph fragment:
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, 3), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_26, 0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_25, %clamp_max_3), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_36, 6), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, 3), kwargs = {})
#   %clamp_min_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_27, 0), kwargs = {})
#   %clamp_max_4 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_4, 6), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_4, 6), kwargs = {})
#   %mul_37 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %div_3), kwargs = {})
triton_poi_fused_hardsigmoid_hardswish_mul_21 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 96)
    x2 = xindex // 1536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 96*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = triton_helpers.maximum(tmp10, tmp3)
    tmp12 = triton_helpers.minimum(tmp11, tmp5)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp13 * tmp7
    tmp15 = tmp8 * tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/me/cmedc44gk6orhz6gk5p54gh2xkiqenahzuucywrjbhu3pgf5e6p2.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_34 => add_29, mul_39, mul_40, sub_11
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_89), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_39, %unsqueeze_93), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_40, %unsqueeze_95), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ns/cnsy4dqsp7qnor3gd4b5fbkk7tlxra5rsf2yhtttk4ekkziz7sob.py
# Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   input_36 => add_31, mul_42, mul_43, sub_12
#   input_37 => add_32, clamp_max_5, clamp_min_5, div_5, mul_44
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_97), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %unsqueeze_101), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_103), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, 3), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_32, 0), kwargs = {})
#   %clamp_max_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 6), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_31, %clamp_max_5), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_44, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3r/c3rgcifpwtpip7rbkvttdipmum62c5s5zrxhuupeqs3gefctve6g.py
# Topologically Sorted Source Nodes: [input_39], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_39 => add_34, mul_46, mul_47, sub_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_105), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_109), kwargs = {})
#   %add_34 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_111), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nv/cnvgpcmdwat6bggee5mfa7teiauutfnzqqm4vgdvkkqu26kfdub6.py
# Topologically Sorted Source Nodes: [input_40, scale_10], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   input_40 => add_35, clamp_max_6, clamp_min_6, div_6, mul_48
#   scale_10 => mean_2
# Graph fragment:
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, 3), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 0), kwargs = {})
#   %clamp_max_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, %clamp_max_6), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_48, 6), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_6, [-1, -2], True), kwargs = {})
triton_per_fused_hardswish_mean_25 = async_compile.triton('triton_per_fused_hardswish_mean_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_25(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 240)
    x1 = xindex // 240
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 240*r2 + 3840*x1), xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 16.0
    tmp15 = tmp13 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y5/cy5mfy5na6bnpeizyphj3yfrpgyfy57xfgyae6ji4bgo33gqfd5d.py
# Topologically Sorted Source Nodes: [scale_11, scale_12], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   scale_11 => convolution_18
#   scale_12 => relu_7
# Graph fragment:
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_2, %primals_80, %primals_81, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_18,), kwargs = {})
triton_poi_fused_convolution_relu_26 = async_compile.triton('triton_poi_fused_convolution_relu_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f7/cf7yzp7crlwf7g7a65ffqdegcsda6p5rruugxhbbpp42tla3mepc.py
# Topologically Sorted Source Nodes: [scale_13], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   scale_13 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_82, %primals_83, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q4/cq4utcgrhsrumx7b3ovwrwgkwysxviybvbtq346z5wsjm4wcfop6.py
# Topologically Sorted Source Nodes: [input_40, scale_14, input_41], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_40 => add_35, clamp_max_6, clamp_min_6, div_6, mul_48
#   input_41 => mul_49
#   scale_14 => add_36, clamp_max_7, clamp_min_7, div_7
# Graph fragment:
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, 3), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_35, 0), kwargs = {})
#   %clamp_max_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_34, %clamp_max_6), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_48, 6), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_19, 3), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_36, 0), kwargs = {})
#   %clamp_max_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 6), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_7, 6), kwargs = {})
#   %mul_49 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_7, %div_6), kwargs = {})
triton_poi_fused_hardsigmoid_hardswish_mul_28 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_28(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 240)
    x2 = xindex // 3840
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 240*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = triton_helpers.maximum(tmp10, tmp3)
    tmp12 = triton_helpers.minimum(tmp11, tmp5)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp13 * tmp7
    tmp15 = tmp8 * tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqs7ikqmmhvze3w7aujmo6syufm53qlyojfja27g3cirn42qipq4.py
# Topologically Sorted Source Nodes: [input_43, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_43 => add_38, mul_51, mul_52, sub_14
#   result_1 => add_39
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_113), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_117), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %unsqueeze_119), kwargs = {})
#   %add_39 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %add_29), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vk/cvkistqzijdts3dwjj6qqg253mpzbeayu3spd6xdqwtaiyxufcg3.py
# Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   input_54 => add_51, mul_66, mul_67, sub_18
#   input_55 => add_52, clamp_max_11, clamp_min_11, div_11, mul_68
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_145), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_66, %unsqueeze_149), kwargs = {})
#   %add_51 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_67, %unsqueeze_151), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_51, 3), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_52, 0), kwargs = {})
#   %clamp_max_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 6), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_51, %clamp_max_11), kwargs = {})
#   %div_11 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_68, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4m/c4mxtakgfpgwjqi545c5spccp4ojkg6hnie7ihhx7vz3ilx7pm5k.py
# Topologically Sorted Source Nodes: [input_57], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_57 => add_54, mul_70, mul_71, sub_19
# Graph fragment:
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_153), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_157), kwargs = {})
#   %add_54 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_159), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvupwx7444dgm4f7srzpxaupd7up7b6onpjtltrvzr5p2hggetym.py
# Topologically Sorted Source Nodes: [input_58, scale_20], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   input_58 => add_55, clamp_max_12, clamp_min_12, div_12, mul_72
#   scale_20 => mean_4
# Graph fragment:
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, 3), kwargs = {})
#   %clamp_min_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_55, 0), kwargs = {})
#   %clamp_max_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, %clamp_max_12), kwargs = {})
#   %div_12 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_72, 6), kwargs = {})
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_12, [-1, -2], True), kwargs = {})
triton_per_fused_hardswish_mean_32 = async_compile.triton('triton_per_fused_hardswish_mean_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_32(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 120)
    x1 = xindex // 120
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 120*r2 + 1920*x1), xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 16.0
    tmp15 = tmp13 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r6/cr6lmxyj6rvsklzfn4ghx5ib7hepd63aqlkflhx2knbj7vppvxio.py
# Topologically Sorted Source Nodes: [scale_21, scale_22], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   scale_21 => convolution_28
#   scale_22 => relu_9
# Graph fragment:
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_4, %primals_118, %primals_119, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_28,), kwargs = {})
triton_poi_fused_convolution_relu_33 = async_compile.triton('triton_poi_fused_convolution_relu_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_33(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vd/cvddhnxfbc4wykmjszef6lwbp3fcjk4au7rps6q3xgo3sf27yzn5.py
# Topologically Sorted Source Nodes: [scale_23], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   scale_23 => convolution_29
# Graph fragment:
#   %convolution_29 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_120, %primals_121, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_34 = async_compile.triton('triton_poi_fused_convolution_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_34(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dc/cdcfo7vtgxu7e2kumxza2gyolmvwuugrcxqxklquq5zpfhuos76t.py
# Topologically Sorted Source Nodes: [input_58, scale_24, input_59], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_58 => add_55, clamp_max_12, clamp_min_12, div_12, mul_72
#   input_59 => mul_73
#   scale_24 => add_56, clamp_max_13, clamp_min_13, div_13
# Graph fragment:
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, 3), kwargs = {})
#   %clamp_min_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_55, 0), kwargs = {})
#   %clamp_max_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_54, %clamp_max_12), kwargs = {})
#   %div_12 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_72, 6), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, 3), kwargs = {})
#   %clamp_min_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_56, 0), kwargs = {})
#   %clamp_max_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_13, 6), kwargs = {})
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_13, 6), kwargs = {})
#   %mul_73 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_13, %div_12), kwargs = {})
triton_poi_fused_hardsigmoid_hardswish_mul_35 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 120)
    x2 = xindex // 1920
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 120*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = triton_helpers.maximum(tmp10, tmp3)
    tmp12 = triton_helpers.minimum(tmp11, tmp5)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp13 * tmp7
    tmp15 = tmp8 * tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d4/cd4vhxaifaqwlqcjh7blb3lfnpeqqejtl4km6xktxhnoejzmrijz.py
# Topologically Sorted Source Nodes: [input_61], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_61 => add_58, mul_75, mul_76, sub_20
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_161), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_75, %unsqueeze_165), kwargs = {})
#   %add_58 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_76, %unsqueeze_167), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/en/cenzuwh3rvlytpfnxyd2ogkdbziehghyki45ccojdignd4nvbefk.py
# Topologically Sorted Source Nodes: [input_63, input_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   input_63 => add_60, mul_78, mul_79, sub_21
#   input_64 => add_61, clamp_max_14, clamp_min_14, div_14, mul_80
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_169), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %unsqueeze_173), kwargs = {})
#   %add_60 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %unsqueeze_175), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, 3), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_61, 0), kwargs = {})
#   %clamp_max_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 6), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %clamp_max_14), kwargs = {})
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_80, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eh/ceh5plgf4wj7yntapfdeuyqyxlkxjildv4mtufdch4fklo7yfzf6.py
# Topologically Sorted Source Nodes: [input_66], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_66 => add_63, mul_82, mul_83, sub_22
# Graph fragment:
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_177), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_181), kwargs = {})
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_183), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wo/cwogdzzc5jm4spoudg363iqojg7n4hiaf2l6qzgrdif23wgmxelt.py
# Topologically Sorted Source Nodes: [input_67, scale_25], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   input_67 => add_64, clamp_max_15, clamp_min_15, div_15, mul_84
#   scale_25 => mean_5
# Graph fragment:
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, 3), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_64, 0), kwargs = {})
#   %clamp_max_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_63, %clamp_max_15), kwargs = {})
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_84, 6), kwargs = {})
#   %mean_5 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_15, [-1, -2], True), kwargs = {})
triton_per_fused_hardswish_mean_39 = async_compile.triton('triton_per_fused_hardswish_mean_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_hardswish_mean_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_hardswish_mean_39(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 144)
    x1 = xindex // 144
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 144*r2 + 2304*x1), xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 16.0
    tmp15 = tmp13 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxvabs7pzfgdw2ihvk4kysonmc4dfgbhvqzuuijapzywhluys2lx.py
# Topologically Sorted Source Nodes: [scale_26, scale_27], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   scale_26 => convolution_33
#   scale_27 => relu_10
# Graph fragment:
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_5, %primals_137, %primals_138, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_33,), kwargs = {})
triton_poi_fused_convolution_relu_40 = async_compile.triton('triton_poi_fused_convolution_relu_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_40(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bb/cbbutjb3kcwvtiaf5n2ix4uu5w2aq4hytlzlsiw4fnzxveh2hw3d.py
# Topologically Sorted Source Nodes: [scale_28], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   scale_28 => convolution_34
# Graph fragment:
#   %convolution_34 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_10, %primals_139, %primals_140, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_41 = async_compile.triton('triton_poi_fused_convolution_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_41(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qj/cqjrj2a32rfeymnpx7opbpuxi5bqlvuqbzajrmvnfx7rxkwypvrg.py
# Topologically Sorted Source Nodes: [input_67, scale_29, input_68], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_67 => add_64, clamp_max_15, clamp_min_15, div_15, mul_84
#   input_68 => mul_85
#   scale_29 => add_65, clamp_max_16, clamp_min_16, div_16
# Graph fragment:
#   %add_64 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, 3), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_64, 0), kwargs = {})
#   %clamp_max_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_63, %clamp_max_15), kwargs = {})
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_84, 6), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_34, 3), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_65, 0), kwargs = {})
#   %clamp_max_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_16, 6), kwargs = {})
#   %mul_85 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %div_15), kwargs = {})
triton_poi_fused_hardsigmoid_hardswish_mul_42 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_42(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 144)
    x2 = xindex // 2304
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 144*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = triton_helpers.maximum(tmp10, tmp3)
    tmp12 = triton_helpers.minimum(tmp11, tmp5)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp13 * tmp7
    tmp15 = tmp8 * tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jg/cjgzgdefaervnu5hib5iqvjbz4lizict4gkh7ysg7pfpokxtvftl.py
# Topologically Sorted Source Nodes: [input_70, result_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_70 => add_67, mul_87, mul_88, sub_23
#   result_3 => add_68
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_185), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_189), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_191), kwargs = {})
#   %add_68 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_67, %add_58), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_43', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3u/c3upjay46evq6dvziejd755n2dnsnr45pabq73536knuggn5bvth.py
# Topologically Sorted Source Nodes: [input_72, input_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   input_72 => add_70, mul_90, mul_91, sub_24
#   input_73 => add_71, clamp_max_17, clamp_min_17, div_17, mul_92
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_36, %unsqueeze_193), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %unsqueeze_197), kwargs = {})
#   %add_70 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %unsqueeze_199), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, 3), kwargs = {})
#   %clamp_min_17 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_71, 0), kwargs = {})
#   %clamp_max_17 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_17, 6), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_70, %clamp_max_17), kwargs = {})
#   %div_17 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_92, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 288)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgdkjdlqggiexi4vhuxdmhwarund5ryfs7noy6glqt3brzotvk7d.py
# Topologically Sorted Source Nodes: [input_75], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_75 => add_73, mul_94, mul_95, sub_25
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_201), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %unsqueeze_205), kwargs = {})
#   %add_73 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_95, %unsqueeze_207), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 288)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/it/citm67yjf2ojh3ikp5fqsnglb5nzefqo5vyw6gaf62ws36avlldv.py
# Topologically Sorted Source Nodes: [input_76, scale_30], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   input_76 => add_74, clamp_max_18, clamp_min_18, div_18, mul_96
#   scale_30 => mean_6
# Graph fragment:
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_73, 3), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_74, 0), kwargs = {})
#   %clamp_max_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, %clamp_max_18), kwargs = {})
#   %div_18 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_96, 6), kwargs = {})
#   %mean_6 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_18, [-1, -2], True), kwargs = {})
triton_poi_fused_hardswish_mean_46 = async_compile.triton('triton_poi_fused_hardswish_mean_46', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_mean_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_mean_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 288)
    x1 = xindex // 288
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1152*x1), xmask)
    tmp10 = tl.load(in_ptr0 + (288 + x0 + 1152*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (576 + x0 + 1152*x1), xmask)
    tmp24 = tl.load(in_ptr0 + (864 + x0 + 1152*x1), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp11, tmp3)
    tmp13 = triton_helpers.minimum(tmp12, tmp5)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 * tmp8
    tmp16 = tmp9 + tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = triton_helpers.minimum(tmp19, tmp5)
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp16 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = triton_helpers.maximum(tmp25, tmp3)
    tmp27 = triton_helpers.minimum(tmp26, tmp5)
    tmp28 = tmp24 * tmp27
    tmp29 = tmp28 * tmp8
    tmp30 = tmp23 + tmp29
    tmp31 = 4.0
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qx/cqx4w7ie3jvo4ss2ht7l7vfrggfj2cj7t6dlukbe65lbaghlgpgn.py
# Topologically Sorted Source Nodes: [scale_31, scale_32], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   scale_31 => convolution_38
#   scale_32 => relu_11
# Graph fragment:
#   %convolution_38 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_6, %primals_156, %primals_157, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_38,), kwargs = {})
triton_poi_fused_convolution_relu_47 = async_compile.triton('triton_poi_fused_convolution_relu_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_47(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 72)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mb/cmbwd5m7mdf7rmrxc5tqakjurpzhde7nphizslmxwrprdrjufd7g.py
# Topologically Sorted Source Nodes: [scale_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   scale_33 => convolution_39
# Graph fragment:
#   %convolution_39 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_158, %primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_48 = async_compile.triton('triton_poi_fused_convolution_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_48(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 288)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s2/cs2iucniypvdikh77lrdd5gkxqw4ksqgqd2k64bxm43gkcithxmm.py
# Topologically Sorted Source Nodes: [input_76, scale_34, input_77], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_76 => add_74, clamp_max_18, clamp_min_18, div_18, mul_96
#   input_77 => mul_97
#   scale_34 => add_75, clamp_max_19, clamp_min_19, div_19
# Graph fragment:
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_73, 3), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_74, 0), kwargs = {})
#   %clamp_max_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_73, %clamp_max_18), kwargs = {})
#   %div_18 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_96, 6), kwargs = {})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_39, 3), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_75, 0), kwargs = {})
#   %clamp_max_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 6), kwargs = {})
#   %div_19 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_19, 6), kwargs = {})
#   %mul_97 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_19, %div_18), kwargs = {})
triton_poi_fused_hardsigmoid_hardswish_mul_49 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_49(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 288)
    x2 = xindex // 1152
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 288*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = triton_helpers.maximum(tmp10, tmp3)
    tmp12 = triton_helpers.minimum(tmp11, tmp5)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp13 * tmp7
    tmp15 = tmp8 * tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/es/cesxnlz2iwfesx4yatkhbjb4s76ppbgi6dg4ypbeg6ze2nzibcti.py
# Topologically Sorted Source Nodes: [input_79], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_79 => add_77, mul_100, mul_99, sub_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_209), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_99, %unsqueeze_213), kwargs = {})
#   %add_77 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_100, %unsqueeze_215), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dr/cdrkgoz3wxico6nt7lgmtvxm2dayca2gtbgmlkwjsvubbkfssb3q.py
# Topologically Sorted Source Nodes: [input_81, input_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# Source node to ATen node mapping:
#   input_81 => add_79, mul_102, mul_103, sub_27
#   input_82 => add_80, clamp_max_20, clamp_min_20, div_20, mul_104
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_217), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %unsqueeze_221), kwargs = {})
#   %add_79 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %unsqueeze_223), kwargs = {})
#   %add_80 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_79, 3), kwargs = {})
#   %clamp_min_20 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_80, 0), kwargs = {})
#   %clamp_max_20 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_20, 6), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_79, %clamp_max_20), kwargs = {})
#   %div_20 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_104, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardswish_51', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardswish_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 576)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rp/crp36cdnrwltui6hkyxoksxnyrj3jwufo7zgppifrxbeaplxeyui.py
# Topologically Sorted Source Nodes: [input_84], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_84 => add_82, mul_106, mul_107, sub_28
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_225), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %unsqueeze_229), kwargs = {})
#   %add_82 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %unsqueeze_231), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 576)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kn/cknef77gsaukfp6ftkjropqexpbfal3xjbz3uep37fq5vefilzq5.py
# Topologically Sorted Source Nodes: [input_85, scale_35], Original ATen: [aten.hardswish, aten.mean]
# Source node to ATen node mapping:
#   input_85 => add_83, clamp_max_21, clamp_min_21, div_21, mul_108
#   scale_35 => mean_7
# Graph fragment:
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, 3), kwargs = {})
#   %clamp_min_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_83, 0), kwargs = {})
#   %clamp_max_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_21, 6), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %clamp_max_21), kwargs = {})
#   %div_21 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_108, 6), kwargs = {})
#   %mean_7 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_21, [-1, -2], True), kwargs = {})
triton_poi_fused_hardswish_mean_53 = async_compile.triton('triton_poi_fused_hardswish_mean_53', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_mean_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardswish_mean_53(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 576)
    x1 = xindex // 576
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2304*x1), xmask)
    tmp10 = tl.load(in_ptr0 + (576 + x0 + 2304*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (1152 + x0 + 2304*x1), xmask)
    tmp24 = tl.load(in_ptr0 + (1728 + x0 + 2304*x1), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp11, tmp3)
    tmp13 = triton_helpers.minimum(tmp12, tmp5)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 * tmp8
    tmp16 = tmp9 + tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = triton_helpers.minimum(tmp19, tmp5)
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp16 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = triton_helpers.maximum(tmp25, tmp3)
    tmp27 = triton_helpers.minimum(tmp26, tmp5)
    tmp28 = tmp24 * tmp27
    tmp29 = tmp28 * tmp8
    tmp30 = tmp23 + tmp29
    tmp31 = 4.0
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z7/cz7edhesmr5643nnb7afudeue6cp6mj6gjcu6fbz3wu5lxtb2vgw.py
# Topologically Sorted Source Nodes: [scale_36, scale_37], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   scale_36 => convolution_43
#   scale_37 => relu_12
# Graph fragment:
#   %convolution_43 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_7, %primals_175, %primals_176, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_43,), kwargs = {})
triton_poi_fused_convolution_relu_54 = async_compile.triton('triton_poi_fused_convolution_relu_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_54(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fp/cfpf24g3ufa5whwc2pmhalnsq7hg4d7odn7rpx2wrovxilzgxgu3.py
# Topologically Sorted Source Nodes: [scale_38], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   scale_38 => convolution_44
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_12, %primals_177, %primals_178, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_55 = async_compile.triton('triton_poi_fused_convolution_55', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_55(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 576)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wh/cwhrddvgbwfxmsd5un4ugj2prlcojqbq62ysef65u5n26f5by72e.py
# Topologically Sorted Source Nodes: [input_85, scale_39, input_86], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
# Source node to ATen node mapping:
#   input_85 => add_83, clamp_max_21, clamp_min_21, div_21, mul_108
#   input_86 => mul_109
#   scale_39 => add_84, clamp_max_22, clamp_min_22, div_22
# Graph fragment:
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, 3), kwargs = {})
#   %clamp_min_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_83, 0), kwargs = {})
#   %clamp_max_21 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_21, 6), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %clamp_max_21), kwargs = {})
#   %div_21 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_108, 6), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_44, 3), kwargs = {})
#   %clamp_min_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_84, 0), kwargs = {})
#   %clamp_max_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_22, 6), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%clamp_max_22, 6), kwargs = {})
#   %mul_109 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_22, %div_21), kwargs = {})
triton_poi_fused_hardsigmoid_hardswish_mul_56 = async_compile.triton('triton_poi_fused_hardsigmoid_hardswish_mul_56', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardsigmoid_hardswish_mul_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_hardsigmoid_hardswish_mul_56(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 576)
    x2 = xindex // 2304
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 576*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = 0.16666666666666666
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 + tmp1
    tmp11 = triton_helpers.maximum(tmp10, tmp3)
    tmp12 = triton_helpers.minimum(tmp11, tmp5)
    tmp13 = tmp9 * tmp12
    tmp14 = tmp13 * tmp7
    tmp15 = tmp8 * tmp14
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5ss2svu5oeqcgugr6ykyjq2e4gjyp7p2qt2njgf2jr5jg3an7q3.py
# Topologically Sorted Source Nodes: [input_88, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_88 => add_86, mul_111, mul_112, sub_29
#   result_4 => add_87
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_233), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_111, %unsqueeze_237), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_112, %unsqueeze_239), kwargs = {})
#   %add_87 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_86, %add_77), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7o/c7oxdlkuqenqh2ts42tugtbkjbqsahj4eutgxz6an5oz3jri45zu.py
# Topologically Sorted Source Nodes: [c3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   c3 => convolution_51
# Graph fragment:
#   %convolution_51 : [num_users=8] = call_function[target=torch.ops.aten.convolution.default](args = (%add_97, %primals_203, %primals_204, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_58 = async_compile.triton('triton_poi_fused_convolution_58', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_58(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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


# kernel path: inductor_cache/vx/cvxgcmcw4dskp7o4wkhn43d7pgr4zx3jwde43sxny2zu67p5i7zj.py
# Topologically Sorted Source Nodes: [input_98, input_99], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_98 => convolution_52
#   input_99 => gt, mul_125, where
# Graph fragment:
#   %convolution_52 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_51, %primals_205, %primals_206, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_52, 0), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_52, 0.01), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_52, %mul_125), kwargs = {})
triton_poi_fused_convolution_leaky_relu_59 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_59', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_59(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/in/cinq6fyifksj4cbip2m5c4g4xzkktyleirgcuhlettwcoi5bnsvj.py
# Topologically Sorted Source Nodes: [mul_9], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_9 => mul_128
# Graph fragment:
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_211, 0.044194173824159216), kwargs = {})
triton_poi_fused_mul_60 = async_compile.triton('triton_poi_fused_mul_60', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_60', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_60(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.044194173824159216
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/53/c536lxtogilluapk6sgmtzpl5iufdq2bpvmm5dyumb2g7okfoi77.py
# Topologically Sorted Source Nodes: [mul_10], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_10 => mul_129
# Graph fragment:
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_212, 1), kwargs = {})
triton_poi_fused_mul_61 = async_compile.triton('triton_poi_fused_mul_61', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_61(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/on/con7tkghysj2qo7gzctgfth7q37cqyi7j3bbgyyevxf46pk6wjgv.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate => convert_element_type_67
# Graph fragment:
#   %convert_element_type_67 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.int64), kwargs = {})
triton_poi_fused__to_copy_62 = async_compile.triton('triton_poi_fused__to_copy_62', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_62', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_62(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ms/cmsfm64tmijx4bthu2vpozydhktiovp65fp24bd4mtqakwcy7xw2.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate => add_101, clamp_max_26
# Graph fragment:
#   %add_101 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_67, 1), kwargs = {})
#   %clamp_max_26 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_101, 1), kwargs = {})
triton_poi_fused_add_clamp_63 = async_compile.triton('triton_poi_fused_add_clamp_63', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_63', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_63(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = triton_helpers.minimum(tmp10, tmp9)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kl/ckl4fljy4ow6htjbsorsgy2en5sr2vizsk2zekvk7qn2f5vr3m5f.py
# Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   interpolate => add_100, clamp_max_28, clamp_min_26, clamp_min_28, convert_element_type_66, iota, mul_140, sub_33, sub_35
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_66 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_66, 0.5), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, 0.5), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_140, 0.5), kwargs = {})
#   %clamp_min_26 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_33, 0.0), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_26, %convert_element_type_69), kwargs = {})
#   %clamp_min_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_35, 0.0), kwargs = {})
#   %clamp_max_28 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_28, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_64 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_64', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_64', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_64(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lr/clrwsndfjghripqojrt3ve3jfai4e24t2awmrqh7qtlmgykifzfy.py
# Topologically Sorted Source Nodes: [conv2d_61, interpolate, p2], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   conv2d_61 => convolution_61
#   interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_104, add_105, add_106, mul_142, mul_143, mul_144, sub_36, sub_37, sub_39
#   p2 => add_107
# Graph fragment:
#   %convolution_61 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_68, %primals_229, %primals_230, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_51, [None, None, %convert_element_type_67, %convert_element_type_69]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_51, [None, None, %convert_element_type_67, %clamp_max_27]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_51, [None, None, %clamp_max_26, %convert_element_type_69]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_51, [None, None, %clamp_max_26, %clamp_max_27]), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %clamp_max_28), kwargs = {})
#   %add_104 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_142), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %clamp_max_28), kwargs = {})
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_143), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_105, %add_104), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %clamp_max_29), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %mul_144), kwargs = {})
#   %add_107 : [num_users=9] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_106, %convolution_61), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sub_65 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sub_65', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sub_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sub_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 4
    x2 = (xindex % 4)
    y0 = (yindex % 512)
    y1 = yindex // 512
    x4 = xindex
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_out_ptr0 + (y0 + 512*x4 + 8192*y1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (y0 + 512*tmp8 + 1024*tmp4 + 2048*y1), xmask)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (y0 + 512*tmp13 + 1024*tmp4 + 2048*y1), xmask)
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (y0 + 512*tmp8 + 1024*tmp22 + 2048*y1), xmask)
    tmp24 = tl.load(in_ptr2 + (y0 + 512*tmp13 + 1024*tmp22 + 2048*y1), xmask)
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp34 = tmp32 + tmp33
    tmp35 = tmp31 + tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + 512*x4 + 8192*y1), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/db/cdblh6wyy4xjd6w33cid7djchauusorjahbfmcrocpvzxxg2yvrk.py
# Topologically Sorted Source Nodes: [input_116, input_117], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_116 => convolution_62
#   input_117 => gt_9, mul_145, where_9
# Graph fragment:
#   %convolution_62 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_107, %primals_231, %primals_232, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_9 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_62, 0), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_62, 0.01), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %convolution_62, %mul_145), kwargs = {})
triton_poi_fused_convolution_leaky_relu_66 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_66', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_66', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_66(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2), tmp4, None)
    tl.store(in_out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/2z/c2zkqp6qgjsigkwl4f7fizazbnrprojw62524hvgr274pw22zkls.py
# Topologically Sorted Source Nodes: [mul_14, out_2, iadd_7, mul_16, out_3, iadd_8], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_7 => add_99
#   iadd_8 => add_108
#   mul_14 => mul_139
#   mul_16 => mul_150
#   out_2 => add_tensor_15
#   out_3 => add_tensor_14
# Graph fragment:
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_228, 1), kwargs = {})
#   %add_tensor_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_15, %mul_139), kwargs = {})
#   %add_99 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_8, %add_tensor_15), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_240, 1), kwargs = {})
#   %add_tensor_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_14, %mul_150), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_16, %add_tensor_14), kwargs = {})
triton_poi_fused_add_addmm_mul_67 = async_compile.triton('triton_poi_fused_add_addmm_mul_67', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_67', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_67(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp4 = tl.load(in_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp29 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 2, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp4 + tmp9
    tmp11 = tl.where(tmp3, tmp10, tmp4)
    tmp12 = tl.where(tmp2, tmp10, tmp4)
    tmp13 = tl.where(tmp2, tmp11, tmp12)
    tmp16 = tmp15 * tmp7
    tmp17 = tmp14 + tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = tl.full([1], 3, tl.int32)
    tmp20 = tmp19 == tmp0
    tmp21 = tmp0 == tmp0
    tmp22 = tl.where(tmp21, tmp18, tmp13)
    tmp23 = tmp19 == tmp1
    tmp24 = tl.where(tmp23, tmp10, tmp4)
    tmp25 = tl.where(tmp23, tmp11, tmp24)
    tmp26 = tl.where(tmp20, tmp18, tmp25)
    tmp27 = tl.where(tmp20, tmp22, tmp26)
    tmp30 = tmp29 * tmp7
    tmp31 = tmp28 + tmp30
    tmp32 = tmp27 + tmp31
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
    tl.store(in_out_ptr1 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxgshkcqhykyjksvlqizzvy2xhe7ask3sild3nbhsmdkk2dp4ni3.py
# Topologically Sorted Source Nodes: [mul_16, out_3, iadd_8], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_8 => add_108
#   mul_16 => mul_150
#   out_3 => add_tensor_14
# Graph fragment:
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_240, 1), kwargs = {})
#   %add_tensor_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_14, %mul_150), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_16, %add_tensor_14), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_28, %add_108, 1, 3), kwargs = {})
triton_poi_fused_add_addmm_mul_68 = async_compile.triton('triton_poi_fused_add_addmm_mul_68', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_68', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_68(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 2, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = tmp4 == tmp4
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp4 == tmp8
    tmp10 = tmp8 == tmp8
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 + tmp15
    tmp17 = tmp11 + tmp16
    tmp18 = tl.where(tmp10, tmp17, tmp11)
    tmp19 = tl.where(tmp9, tmp17, tmp11)
    tmp20 = tl.where(tmp9, tmp18, tmp19)
    tmp21 = tl.where(tmp6, tmp7, tmp20)
    tmp22 = tmp0 == tmp8
    tmp23 = tl.where(tmp22, tmp17, tmp11)
    tmp24 = tl.where(tmp22, tmp18, tmp23)
    tmp25 = tl.where(tmp5, tmp7, tmp24)
    tmp26 = tl.where(tmp5, tmp21, tmp25)
    tmp27 = tl.where(tmp2, tmp3, tmp26)
    tl.store(out_ptr0 + (x3), tmp27, None)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5irftuiyozgyrmzvpyrttt2cdgqydliiql2oes5bxlrm47yhwbr.py
# Topologically Sorted Source Nodes: [mul_18, out_4, iadd_9], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_9 => add_109
#   mul_18 => mul_156
#   out_4 => add_tensor_13
# Graph fragment:
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_250, 1), kwargs = {})
#   %add_tensor_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_13, %mul_156), kwargs = {})
#   %add_109 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_24, %add_tensor_13), kwargs = {})
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_41, %add_109, 1, 4), kwargs = {})
triton_poi_fused_add_addmm_mul_69 = async_compile.triton('triton_poi_fused_add_addmm_mul_69', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_69(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (1536 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2048 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 4, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 3, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/va/cvaxeipgiyb5j7z5lbcyqjen3xxhyi5ipkbntdqxbx2jvhegdtag.py
# Topologically Sorted Source Nodes: [mul_20, out_5, iadd_10], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_10 => add_110
#   mul_20 => mul_162
#   out_5 => add_tensor_12
# Graph fragment:
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_260, 1), kwargs = {})
#   %add_tensor_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_12, %mul_162), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_32, %add_tensor_12), kwargs = {})
#   %select_scatter_default_8 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_54, %add_110, 1, 5), kwargs = {})
triton_poi_fused_add_addmm_mul_70 = async_compile.triton('triton_poi_fused_add_addmm_mul_70', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_70', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_70(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (2048 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2560 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 5, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 4, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/rh/crhpr6ummsjitd4h6ijbvqejqchekgbxo4ivfldsa337x3nqlezo.py
# Topologically Sorted Source Nodes: [mul_22, out_6, iadd_11], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_11 => add_111
#   mul_22 => mul_168
#   out_6 => add_tensor_11
# Graph fragment:
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_270, 1), kwargs = {})
#   %add_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_11, %mul_168), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_40, %add_tensor_11), kwargs = {})
#   %select_scatter_default_10 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_67, %add_111, 1, 6), kwargs = {})
triton_poi_fused_add_addmm_mul_71 = async_compile.triton('triton_poi_fused_add_addmm_mul_71', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_71', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_71(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (2560 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3072 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 6, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 5, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/67/c673zfo364ipmdoa56kubhzzxl42d5ajat3x6sbmfslko4nxrltd.py
# Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   interpolate_1 => convert_element_type_71
# Graph fragment:
#   %convert_element_type_71 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.int64), kwargs = {})
triton_poi_fused__to_copy_72 = async_compile.triton('triton_poi_fused__to_copy_72', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_72', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_72(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3q/c3qqhg7a6psyhctacum6qdgdrvk3plo6jofedphm63jqjxkfy2io.py
# Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   interpolate_1 => add_113, clamp_max_30
# Graph fragment:
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_71, 1), kwargs = {})
#   %clamp_max_30 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_113, 3), kwargs = {})
triton_poi_fused_add_clamp_73 = async_compile.triton('triton_poi_fused_add_clamp_73', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_73', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_73(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/py/cpyhosroyrbso3l7axelgtxhsskqmuvivj6uoe6kzq77u52bfzhs.py
# Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   interpolate_1 => add_112, clamp_max_32, clamp_min_30, clamp_min_32, convert_element_type_70, iota_2, mul_169, sub_40, sub_42
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_70 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %add_112 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_70, 0.5), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_112, 0.5), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_169, 0.5), kwargs = {})
#   %clamp_min_30 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_40, 0.0), kwargs = {})
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_30, %convert_element_type_73), kwargs = {})
#   %clamp_min_32 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_42, 0.0), kwargs = {})
#   %clamp_max_32 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_32, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_74 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_74', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_74', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_74(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3 * tmp2
    tmp5 = tmp4 - tmp2
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 - tmp9
    tmp11 = triton_helpers.maximum(tmp10, tmp6)
    tmp12 = 1.0
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yy/cyyiayghr7lpy7trr2mh3ndo5lc5awkgo6uazcdkjxyh4te7dgxv.py
# Topologically Sorted Source Nodes: [conv2d_78, interpolate_1, p1], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   conv2d_78 => convolution_78
#   interpolate_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_116, add_117, add_118, mul_171, mul_172, mul_173, sub_43, sub_44, sub_46
#   p1 => add_119
# Graph fragment:
#   %convolution_78 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_20, %primals_271, %primals_272, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_107, [None, None, %convert_element_type_71, %convert_element_type_73]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_107, [None, None, %convert_element_type_71, %clamp_max_31]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_107, [None, None, %clamp_max_30, %convert_element_type_73]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_107, [None, None, %clamp_max_30, %clamp_max_31]), kwargs = {})
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %clamp_max_32), kwargs = {})
#   %add_116 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_171), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %clamp_max_32), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_172), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_117, %add_116), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %clamp_max_33), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_116, %mul_173), kwargs = {})
#   %add_119 : [num_users=12] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %convolution_78), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_mul_sub_75 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_mul_sub_75', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_mul_sub_75', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_mul_sub_75(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex // 8
    x2 = (xindex % 8)
    y0 = (yindex % 512)
    y1 = yindex // 512
    x4 = xindex
    y5 = yindex
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x3), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_out_ptr0 + (y0 + 512*x4 + 32768*y1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr7 + (y0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (y0 + 512*tmp8 + 2048*tmp4 + 8192*y1), xmask)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (y0 + 512*tmp13 + 2048*tmp4 + 8192*y1), xmask)
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (y0 + 512*tmp8 + 2048*tmp22 + 8192*y1), xmask)
    tmp24 = tl.load(in_ptr2 + (y0 + 512*tmp13 + 2048*tmp22 + 8192*y1), xmask)
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp34 = tmp32 + tmp33
    tmp35 = tmp31 + tmp34
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + 512*x4 + 32768*y1), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxugouuqdy4mabqwlhhiuaz2kdz3i2p6vgloccxqvkggmcd3myw.py
# Topologically Sorted Source Nodes: [input_148, input_149], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_148 => convolution_79
#   input_149 => gt_25, mul_174, where_25
# Graph fragment:
#   %convolution_79 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%add_119, %primals_273, %primals_274, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_25 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_79, 0), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_79, 0.01), kwargs = {})
#   %where_25 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %convolution_79, %mul_174), kwargs = {})
triton_poi_fused_convolution_leaky_relu_76 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_76', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_76', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_76(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(out_ptr0 + (x2), tmp4, None)
    tl.store(in_out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqbg3zl32mdoyj2hloxwpvjq4rl2ur63y7p6kfn4zdkcwiukhr4a.py
# Topologically Sorted Source Nodes: [mul_24, out_7, iadd_12], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_12 => add_120
#   mul_24 => mul_180
#   out_7 => add_tensor_10
# Graph fragment:
#   %mul_180 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_284, 1), kwargs = {})
#   %add_tensor_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %mul_180), kwargs = {})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_48, %add_tensor_10), kwargs = {})
#   %select_scatter_default_12 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_80, %add_120, 1, 7), kwargs = {})
triton_poi_fused_add_addmm_mul_77 = async_compile.triton('triton_poi_fused_add_addmm_mul_77', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_77', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_77(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (3072 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3584 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 7, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 6, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqslhpb7lrcqv62wjkwexmxczqsegaxyfptvnqaotfvce3ksxbgl.py
# Topologically Sorted Source Nodes: [mul_26, out_8, iadd_13], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_13 => add_121
#   mul_26 => mul_187
#   out_8 => add_tensor_9
# Graph fragment:
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_296, 1), kwargs = {})
#   %add_tensor_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_9, %mul_187), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_56, %add_tensor_9), kwargs = {})
#   %select_scatter_default_14 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_93, %add_121, 1, 8), kwargs = {})
triton_poi_fused_add_addmm_mul_78 = async_compile.triton('triton_poi_fused_add_addmm_mul_78', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_78', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_78(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (3584 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (4096 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 8, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 7, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/qp/cqpugsymtu63obcjo6zijic6ap72s32rabpnuiqdivhkxh25dtsz.py
# Topologically Sorted Source Nodes: [mul_28, out_9, iadd_14], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_14 => add_122
#   mul_28 => mul_194
#   out_9 => add_tensor_8
# Graph fragment:
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_308, 1), kwargs = {})
#   %add_tensor_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_8, %mul_194), kwargs = {})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_64, %add_tensor_8), kwargs = {})
#   %select_scatter_default_16 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_106, %add_122, 1, 9), kwargs = {})
triton_poi_fused_add_addmm_mul_79 = async_compile.triton('triton_poi_fused_add_addmm_mul_79', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_79', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_79(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (4096 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (4608 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 9, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 8, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/lt/clt7az2ixfxurbpf5scwwvtt3xq6oky2fczq77f7ua4i657fnr7u.py
# Topologically Sorted Source Nodes: [mul_30, out_10, iadd_15], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_15 => add_123
#   mul_30 => mul_201
#   out_10 => add_tensor_7
# Graph fragment:
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_320, 1), kwargs = {})
#   %add_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_7, %mul_201), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_72, %add_tensor_7), kwargs = {})
#   %select_scatter_default_18 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_119, %add_123, 1, 10), kwargs = {})
triton_poi_fused_add_addmm_mul_80 = async_compile.triton('triton_poi_fused_add_addmm_mul_80', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_80', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_80(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (4608 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (5120 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 10, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 9, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/zz/czzkba5aebypoisvn2p6w4hmnqwnmzt62y5qcu7elv3cbm53cvix.py
# Topologically Sorted Source Nodes: [mul_32, out_11, iadd_16], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_16 => add_124
#   mul_32 => mul_208
#   out_11 => add_tensor_6
# Graph fragment:
#   %mul_208 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_332, 1), kwargs = {})
#   %add_tensor_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %mul_208), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_80, %add_tensor_6), kwargs = {})
#   %select_scatter_default_20 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_132, %add_124, 1, 11), kwargs = {})
triton_poi_fused_add_addmm_mul_81 = async_compile.triton('triton_poi_fused_add_addmm_mul_81', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_81', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_81(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (5120 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (5632 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 11, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 10, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/5w/c5wq766vt64m5proe3da5cph5o7se3gltjdbvyxq4xvrmhgeqgo6.py
# Topologically Sorted Source Nodes: [mul_34, out_12, iadd_17], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_17 => add_125
#   mul_34 => mul_215
#   out_12 => add_tensor_5
# Graph fragment:
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_344, 1), kwargs = {})
#   %add_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_5, %mul_215), kwargs = {})
#   %add_125 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_88, %add_tensor_5), kwargs = {})
#   %select_scatter_default_22 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_145, %add_125, 1, 12), kwargs = {})
triton_poi_fused_add_addmm_mul_82 = async_compile.triton('triton_poi_fused_add_addmm_mul_82', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_82', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_82(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (5632 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (6144 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 12, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 11, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ze/czeqnzxqxcbytewnjrgktlepbbvv46etxdrsotvssgr43jdkgqv6.py
# Topologically Sorted Source Nodes: [mul_36, out_13, iadd_18], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_18 => add_126
#   mul_36 => mul_222
#   out_13 => add_tensor_4
# Graph fragment:
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_356, 1), kwargs = {})
#   %add_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %mul_222), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_96, %add_tensor_4), kwargs = {})
#   %select_scatter_default_24 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_158, %add_126, 1, 13), kwargs = {})
triton_poi_fused_add_addmm_mul_83 = async_compile.triton('triton_poi_fused_add_addmm_mul_83', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_83', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_83(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (6144 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (6656 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 13, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 12, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/jq/cjqw3lv5jzytetsffqgb3o4kou5kkwu6j42yxsespdjdc5whfshh.py
# Topologically Sorted Source Nodes: [mul_38, out_14, iadd_19], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_19 => add_127
#   mul_38 => mul_229
#   out_14 => add_tensor_3
# Graph fragment:
#   %mul_229 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_368, 1), kwargs = {})
#   %add_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_3, %mul_229), kwargs = {})
#   %add_127 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_104, %add_tensor_3), kwargs = {})
#   %select_scatter_default_26 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_171, %add_127, 1, 14), kwargs = {})
triton_poi_fused_add_addmm_mul_84 = async_compile.triton('triton_poi_fused_add_addmm_mul_84', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_84', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_84(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (6656 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (7168 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 14, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 13, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/3e/c3eilpdljupbkibzwpbhgdcmhalahgwdcip3fwdzhstshdagxhyo.py
# Topologically Sorted Source Nodes: [mul_40, out_15, iadd_20], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_20 => add_128
#   mul_40 => mul_236
#   out_15 => add_tensor_2
# Graph fragment:
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_380, 1), kwargs = {})
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %mul_236), kwargs = {})
#   %add_128 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_112, %add_tensor_2), kwargs = {})
#   %select_scatter_default_28 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_184, %add_128, 1, 15), kwargs = {})
triton_poi_fused_add_addmm_mul_85 = async_compile.triton('triton_poi_fused_add_addmm_mul_85', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_85', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_85(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (7168 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (7680 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 15, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 14, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/6l/c6l5wwjqdbl45fwcps246nmufm3btsgmslxv5sahkna35um54zi2.py
# Topologically Sorted Source Nodes: [mul_42, out_16, iadd_21], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_21 => add_129
#   mul_42 => mul_243
#   out_16 => add_tensor_1
# Graph fragment:
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_392, 1), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %mul_243), kwargs = {})
#   %add_129 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_120, %add_tensor_1), kwargs = {})
#   %select_scatter_default_30 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_197, %add_129, 1, 16), kwargs = {})
triton_poi_fused_add_addmm_mul_86 = async_compile.triton('triton_poi_fused_add_addmm_mul_86', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_86', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_86(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (7680 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (8192 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 16, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 15, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/tw/ctw4wvfrc77wyrlqh7x3fl5xpbif3k6k3cslxp7mmey3k3nabqsg.py
# Topologically Sorted Source Nodes: [mul_44, out_17, iadd_22], Original ATen: [aten.mul, aten.addmm, aten.add]
# Source node to ATen node mapping:
#   iadd_22 => add_130
#   mul_44 => mul_250
#   out_17 => add_tensor
# Graph fragment:
#   %mul_250 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_404, 1), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %mul_250), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_128, %add_tensor), kwargs = {})
#   %select_scatter_default_32 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_210, %add_130, 1, 17), kwargs = {})
triton_poi_fused_add_addmm_mul_87 = async_compile.triton('triton_poi_fused_add_addmm_mul_87', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_mul_87', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_mul_87(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    tmp5 = tl.load(in_ptr0 + (8192 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (8704 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (x3), None)
    tmp0 = x1
    tmp1 = tl.full([1], 17, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1], 16, tl.int32)
    tmp4 = tmp1 == tmp3
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tmp7 + tmp12
    tmp14 = tmp0 == tmp3
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tl.where(tmp2, tmp13, tmp16)
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/vk/cvkceumaqhmee4now3rjf4rsyhlym6j53f2ei5byig57tny2amby.py
# Topologically Sorted Source Nodes: [repeat_1, w_1], Original ATen: [aten.repeat, aten.add]
# Source node to ATen node mapping:
#   repeat_1 => repeat_1
#   w_1 => add_131
# Graph fragment:
#   %select_scatter_default_33 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%permute_213, %select_129, 1, 17), kwargs = {})
#   %repeat_1 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_405, [4, 1, 1]), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_scatter_default_33, %repeat_1), kwargs = {})
triton_poi_fused_add_repeat_88 = async_compile.triton('triton_poi_fused_add_repeat_88', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_repeat_88', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_repeat_88(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 512) % 18)
    x0 = (xindex % 512)
    x2 = xindex // 9216
    x3 = xindex
    x4 = (xindex % 9216)
    tmp3 = tl.load(in_ptr0 + (8704 + x0 + 9216*x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (x3), None)
    tmp6 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 17, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x0 + 512*x2 + 2048*x1), tmp7, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (16, ), (1, ))
    assert_size_stride(primals_12, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_17, (16, ), (1, ))
    assert_size_stride(primals_18, (16, ), (1, ))
    assert_size_stride(primals_19, (16, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (72, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_22, (72, ), (1, ))
    assert_size_stride(primals_23, (72, ), (1, ))
    assert_size_stride(primals_24, (72, ), (1, ))
    assert_size_stride(primals_25, (72, ), (1, ))
    assert_size_stride(primals_26, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (72, ), (1, ))
    assert_size_stride(primals_28, (72, ), (1, ))
    assert_size_stride(primals_29, (72, ), (1, ))
    assert_size_stride(primals_30, (72, ), (1, ))
    assert_size_stride(primals_31, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_32, (24, ), (1, ))
    assert_size_stride(primals_33, (24, ), (1, ))
    assert_size_stride(primals_34, (24, ), (1, ))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_36, (88, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_37, (88, ), (1, ))
    assert_size_stride(primals_38, (88, ), (1, ))
    assert_size_stride(primals_39, (88, ), (1, ))
    assert_size_stride(primals_40, (88, ), (1, ))
    assert_size_stride(primals_41, (88, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_42, (88, ), (1, ))
    assert_size_stride(primals_43, (88, ), (1, ))
    assert_size_stride(primals_44, (88, ), (1, ))
    assert_size_stride(primals_45, (88, ), (1, ))
    assert_size_stride(primals_46, (24, 88, 1, 1), (88, 1, 1, 1))
    assert_size_stride(primals_47, (24, ), (1, ))
    assert_size_stride(primals_48, (24, ), (1, ))
    assert_size_stride(primals_49, (24, ), (1, ))
    assert_size_stride(primals_50, (24, ), (1, ))
    assert_size_stride(primals_51, (96, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_52, (96, ), (1, ))
    assert_size_stride(primals_53, (96, ), (1, ))
    assert_size_stride(primals_54, (96, ), (1, ))
    assert_size_stride(primals_55, (96, ), (1, ))
    assert_size_stride(primals_56, (96, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_57, (96, ), (1, ))
    assert_size_stride(primals_58, (96, ), (1, ))
    assert_size_stride(primals_59, (96, ), (1, ))
    assert_size_stride(primals_60, (96, ), (1, ))
    assert_size_stride(primals_61, (24, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_62, (24, ), (1, ))
    assert_size_stride(primals_63, (96, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_64, (96, ), (1, ))
    assert_size_stride(primals_65, (40, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_66, (40, ), (1, ))
    assert_size_stride(primals_67, (40, ), (1, ))
    assert_size_stride(primals_68, (40, ), (1, ))
    assert_size_stride(primals_69, (40, ), (1, ))
    assert_size_stride(primals_70, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_71, (240, ), (1, ))
    assert_size_stride(primals_72, (240, ), (1, ))
    assert_size_stride(primals_73, (240, ), (1, ))
    assert_size_stride(primals_74, (240, ), (1, ))
    assert_size_stride(primals_75, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_76, (240, ), (1, ))
    assert_size_stride(primals_77, (240, ), (1, ))
    assert_size_stride(primals_78, (240, ), (1, ))
    assert_size_stride(primals_79, (240, ), (1, ))
    assert_size_stride(primals_80, (64, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (240, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_83, (240, ), (1, ))
    assert_size_stride(primals_84, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_85, (40, ), (1, ))
    assert_size_stride(primals_86, (40, ), (1, ))
    assert_size_stride(primals_87, (40, ), (1, ))
    assert_size_stride(primals_88, (40, ), (1, ))
    assert_size_stride(primals_89, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_90, (240, ), (1, ))
    assert_size_stride(primals_91, (240, ), (1, ))
    assert_size_stride(primals_92, (240, ), (1, ))
    assert_size_stride(primals_93, (240, ), (1, ))
    assert_size_stride(primals_94, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_95, (240, ), (1, ))
    assert_size_stride(primals_96, (240, ), (1, ))
    assert_size_stride(primals_97, (240, ), (1, ))
    assert_size_stride(primals_98, (240, ), (1, ))
    assert_size_stride(primals_99, (64, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_100, (64, ), (1, ))
    assert_size_stride(primals_101, (240, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_102, (240, ), (1, ))
    assert_size_stride(primals_103, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_104, (40, ), (1, ))
    assert_size_stride(primals_105, (40, ), (1, ))
    assert_size_stride(primals_106, (40, ), (1, ))
    assert_size_stride(primals_107, (40, ), (1, ))
    assert_size_stride(primals_108, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_109, (120, ), (1, ))
    assert_size_stride(primals_110, (120, ), (1, ))
    assert_size_stride(primals_111, (120, ), (1, ))
    assert_size_stride(primals_112, (120, ), (1, ))
    assert_size_stride(primals_113, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_114, (120, ), (1, ))
    assert_size_stride(primals_115, (120, ), (1, ))
    assert_size_stride(primals_116, (120, ), (1, ))
    assert_size_stride(primals_117, (120, ), (1, ))
    assert_size_stride(primals_118, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_119, (32, ), (1, ))
    assert_size_stride(primals_120, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_121, (120, ), (1, ))
    assert_size_stride(primals_122, (48, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_123, (48, ), (1, ))
    assert_size_stride(primals_124, (48, ), (1, ))
    assert_size_stride(primals_125, (48, ), (1, ))
    assert_size_stride(primals_126, (48, ), (1, ))
    assert_size_stride(primals_127, (144, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_128, (144, ), (1, ))
    assert_size_stride(primals_129, (144, ), (1, ))
    assert_size_stride(primals_130, (144, ), (1, ))
    assert_size_stride(primals_131, (144, ), (1, ))
    assert_size_stride(primals_132, (144, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_133, (144, ), (1, ))
    assert_size_stride(primals_134, (144, ), (1, ))
    assert_size_stride(primals_135, (144, ), (1, ))
    assert_size_stride(primals_136, (144, ), (1, ))
    assert_size_stride(primals_137, (40, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_138, (40, ), (1, ))
    assert_size_stride(primals_139, (144, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_140, (144, ), (1, ))
    assert_size_stride(primals_141, (48, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_142, (48, ), (1, ))
    assert_size_stride(primals_143, (48, ), (1, ))
    assert_size_stride(primals_144, (48, ), (1, ))
    assert_size_stride(primals_145, (48, ), (1, ))
    assert_size_stride(primals_146, (288, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_147, (288, ), (1, ))
    assert_size_stride(primals_148, (288, ), (1, ))
    assert_size_stride(primals_149, (288, ), (1, ))
    assert_size_stride(primals_150, (288, ), (1, ))
    assert_size_stride(primals_151, (288, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_152, (288, ), (1, ))
    assert_size_stride(primals_153, (288, ), (1, ))
    assert_size_stride(primals_154, (288, ), (1, ))
    assert_size_stride(primals_155, (288, ), (1, ))
    assert_size_stride(primals_156, (72, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_157, (72, ), (1, ))
    assert_size_stride(primals_158, (288, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_159, (288, ), (1, ))
    assert_size_stride(primals_160, (96, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_161, (96, ), (1, ))
    assert_size_stride(primals_162, (96, ), (1, ))
    assert_size_stride(primals_163, (96, ), (1, ))
    assert_size_stride(primals_164, (96, ), (1, ))
    assert_size_stride(primals_165, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_166, (576, ), (1, ))
    assert_size_stride(primals_167, (576, ), (1, ))
    assert_size_stride(primals_168, (576, ), (1, ))
    assert_size_stride(primals_169, (576, ), (1, ))
    assert_size_stride(primals_170, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_171, (576, ), (1, ))
    assert_size_stride(primals_172, (576, ), (1, ))
    assert_size_stride(primals_173, (576, ), (1, ))
    assert_size_stride(primals_174, (576, ), (1, ))
    assert_size_stride(primals_175, (144, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_176, (144, ), (1, ))
    assert_size_stride(primals_177, (576, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_178, (576, ), (1, ))
    assert_size_stride(primals_179, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_180, (96, ), (1, ))
    assert_size_stride(primals_181, (96, ), (1, ))
    assert_size_stride(primals_182, (96, ), (1, ))
    assert_size_stride(primals_183, (96, ), (1, ))
    assert_size_stride(primals_184, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_185, (576, ), (1, ))
    assert_size_stride(primals_186, (576, ), (1, ))
    assert_size_stride(primals_187, (576, ), (1, ))
    assert_size_stride(primals_188, (576, ), (1, ))
    assert_size_stride(primals_189, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_190, (576, ), (1, ))
    assert_size_stride(primals_191, (576, ), (1, ))
    assert_size_stride(primals_192, (576, ), (1, ))
    assert_size_stride(primals_193, (576, ), (1, ))
    assert_size_stride(primals_194, (144, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_195, (144, ), (1, ))
    assert_size_stride(primals_196, (576, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_197, (576, ), (1, ))
    assert_size_stride(primals_198, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_199, (96, ), (1, ))
    assert_size_stride(primals_200, (96, ), (1, ))
    assert_size_stride(primals_201, (96, ), (1, ))
    assert_size_stride(primals_202, (96, ), (1, ))
    assert_size_stride(primals_203, (512, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (512, 512), (512, 1))
    assert_size_stride(primals_212, (512, ), (1, ))
    assert_size_stride(primals_213, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_216, (512, ), (1, ))
    assert_size_stride(primals_217, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_218, (512, ), (1, ))
    assert_size_stride(primals_219, (512, 512), (512, 1))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_225, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (512, 512), (512, 1))
    assert_size_stride(primals_228, (512, ), (1, ))
    assert_size_stride(primals_229, (512, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_230, (512, ), (1, ))
    assert_size_stride(primals_231, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_236, (512, ), (1, ))
    assert_size_stride(primals_237, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_239, (512, 512), (512, 1))
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_242, (512, ), (1, ))
    assert_size_stride(primals_243, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_249, (512, 512), (512, 1))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_252, (512, ), (1, ))
    assert_size_stride(primals_253, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_254, (512, ), (1, ))
    assert_size_stride(primals_255, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (512, 512), (512, 1))
    assert_size_stride(primals_260, (512, ), (1, ))
    assert_size_stride(primals_261, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_262, (512, ), (1, ))
    assert_size_stride(primals_263, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_267, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, 512), (512, 1))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (512, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_272, (512, ), (1, ))
    assert_size_stride(primals_273, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_279, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_280, (512, ), (1, ))
    assert_size_stride(primals_281, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_282, (512, ), (1, ))
    assert_size_stride(primals_283, (512, 512), (512, 1))
    assert_size_stride(primals_284, (512, ), (1, ))
    assert_size_stride(primals_285, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (512, 512), (512, 1))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_297, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_302, (512, ), (1, ))
    assert_size_stride(primals_303, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, 512), (512, 1))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_309, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_314, (512, ), (1, ))
    assert_size_stride(primals_315, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_318, (512, ), (1, ))
    assert_size_stride(primals_319, (512, 512), (512, 1))
    assert_size_stride(primals_320, (512, ), (1, ))
    assert_size_stride(primals_321, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_324, (512, ), (1, ))
    assert_size_stride(primals_325, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_326, (512, ), (1, ))
    assert_size_stride(primals_327, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_328, (512, ), (1, ))
    assert_size_stride(primals_329, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_330, (512, ), (1, ))
    assert_size_stride(primals_331, (512, 512), (512, 1))
    assert_size_stride(primals_332, (512, ), (1, ))
    assert_size_stride(primals_333, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_334, (512, ), (1, ))
    assert_size_stride(primals_335, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_336, (512, ), (1, ))
    assert_size_stride(primals_337, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_338, (512, ), (1, ))
    assert_size_stride(primals_339, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_342, (512, ), (1, ))
    assert_size_stride(primals_343, (512, 512), (512, 1))
    assert_size_stride(primals_344, (512, ), (1, ))
    assert_size_stride(primals_345, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_346, (512, ), (1, ))
    assert_size_stride(primals_347, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_348, (512, ), (1, ))
    assert_size_stride(primals_349, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_350, (512, ), (1, ))
    assert_size_stride(primals_351, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_352, (512, ), (1, ))
    assert_size_stride(primals_353, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (512, 512), (512, 1))
    assert_size_stride(primals_356, (512, ), (1, ))
    assert_size_stride(primals_357, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_358, (512, ), (1, ))
    assert_size_stride(primals_359, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_360, (512, ), (1, ))
    assert_size_stride(primals_361, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_362, (512, ), (1, ))
    assert_size_stride(primals_363, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_364, (512, ), (1, ))
    assert_size_stride(primals_365, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_366, (512, ), (1, ))
    assert_size_stride(primals_367, (512, 512), (512, 1))
    assert_size_stride(primals_368, (512, ), (1, ))
    assert_size_stride(primals_369, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_370, (512, ), (1, ))
    assert_size_stride(primals_371, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_372, (512, ), (1, ))
    assert_size_stride(primals_373, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_374, (512, ), (1, ))
    assert_size_stride(primals_375, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_376, (512, ), (1, ))
    assert_size_stride(primals_377, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_378, (512, ), (1, ))
    assert_size_stride(primals_379, (512, 512), (512, 1))
    assert_size_stride(primals_380, (512, ), (1, ))
    assert_size_stride(primals_381, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_382, (512, ), (1, ))
    assert_size_stride(primals_383, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_384, (512, ), (1, ))
    assert_size_stride(primals_385, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_386, (512, ), (1, ))
    assert_size_stride(primals_387, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_388, (512, ), (1, ))
    assert_size_stride(primals_389, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_390, (512, ), (1, ))
    assert_size_stride(primals_391, (512, 512), (512, 1))
    assert_size_stride(primals_392, (512, ), (1, ))
    assert_size_stride(primals_393, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_394, (512, ), (1, ))
    assert_size_stride(primals_395, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_396, (512, ), (1, ))
    assert_size_stride(primals_397, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_398, (512, ), (1, ))
    assert_size_stride(primals_399, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_400, (512, ), (1, ))
    assert_size_stride(primals_401, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_402, (512, ), (1, ))
    assert_size_stride(primals_403, (512, 512), (512, 1))
    assert_size_stride(primals_404, (512, ), (1, ))
    assert_size_stride(primals_405, (18, 512), (512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_205, buf2, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_205
        buf3 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_207, buf3, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_207
        buf4 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_209, buf4, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_209
        buf5 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_213, buf5, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_213
        buf6 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_215, buf6, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_215
        buf7 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_217, buf7, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_217
        buf8 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_221, buf8, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_221
        buf9 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_223, buf9, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_223
        buf10 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_225, buf10, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_225
        buf11 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_231, buf11, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_231
        buf12 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_233, buf12, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_233
        buf13 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_235, buf13, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_235
        buf14 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_237, buf14, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_237
        buf15 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_241, buf15, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_241
        buf16 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_243, buf16, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_243
        buf17 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_245, buf17, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_245
        buf18 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_247, buf18, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_247
        buf19 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_251, buf19, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_251
        buf20 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_253, buf20, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_253
        buf21 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_255, buf21, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_255
        buf22 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_257, buf22, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_257
        buf23 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_261, buf23, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_261
        buf24 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_263, buf24, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_263
        buf25 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_265, buf25, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_265
        buf26 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_267, buf26, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_267
        buf27 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_273, buf27, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_273
        buf28 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_275, buf28, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_275
        buf29 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_277, buf29, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_277
        buf30 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_279, buf30, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_279
        buf31 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_281, buf31, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_281
        buf32 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_285, buf32, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_285
        buf33 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_287, buf33, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_287
        buf34 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_289, buf34, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_289
        buf35 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_291, buf35, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_291
        buf36 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_293, buf36, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_293
        buf37 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_297, buf37, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_297
        buf38 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_299, buf38, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_299
        buf39 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_301, buf39, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_301
        buf40 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_303, buf40, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_303
        buf41 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_305, buf41, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_305
        buf42 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_309, buf42, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_309
        buf43 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_311, buf43, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_311
        buf44 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_313, buf44, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_313
        buf45 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_315, buf45, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_315
        buf46 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_317, buf46, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_317
        buf47 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_321, buf47, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_321
        buf48 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_323, buf48, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_323
        buf49 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_325, buf49, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_325
        buf50 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_327, buf50, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_327
        buf51 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_329, buf51, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_329
        buf52 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_333, buf52, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_333
        buf53 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_335, buf53, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_335
        buf54 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_337, buf54, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_337
        buf55 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_339, buf55, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_339
        buf56 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_341, buf56, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_341
        buf57 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_345, buf57, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_345
        buf58 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_347, buf58, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_347
        buf59 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_349, buf59, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_349
        buf60 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_351, buf60, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_351
        buf61 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_353, buf61, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_353
        buf62 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_357, buf62, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_357
        buf63 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_359, buf63, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_359
        buf64 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_361, buf64, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_361
        buf65 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_363, buf65, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_363
        buf66 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_365, buf66, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_365
        buf67 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_369, buf67, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_369
        buf68 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_371, buf68, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_371
        buf69 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_373, buf69, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_373
        buf70 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_375, buf70, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_375
        buf71 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_377, buf71, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_377
        buf72 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_381, buf72, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_381
        buf73 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_383, buf73, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_383
        buf74 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_385, buf74, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_385
        buf75 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_387, buf75, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_387
        buf76 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_389, buf76, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_389
        buf77 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_393, buf77, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_393
        buf78 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_395, buf78, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_395
        buf79 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_397, buf79, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_397
        buf80 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_399, buf80, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_399
        buf81 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_401, buf81, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 16, 32, 32), (16384, 1, 512, 16))
        buf83 = empty_strided_cuda((4, 16, 32, 32), (16384, 1, 512, 16), torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3.run(buf84, buf82, primals_3, primals_4, primals_5, primals_6, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf85, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf86 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf85, primals_8, primals_9, primals_10, primals_11, buf86, 16384, grid=grid(16384), stream=stream0)
        buf87 = empty_strided_cuda((4, 16, 1, 1, 2), (32, 1, 128, 128, 16), torch.float32)
        # Topologically Sorted Source Nodes: [scale], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_5.run(buf86, buf87, 128, 128, grid=grid(128), stream=stream0)
        buf88 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf89 = reinterpret_tensor(buf88, (4, 16, 1, 1), (16, 1, 16, 16), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [scale], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_6.run(buf89, buf87, 64, 2, grid=grid(64), stream=stream0)
        del buf87
        # Topologically Sorted Source Nodes: [scale_1], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 8, 1, 1), (8, 1, 8, 8))
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [scale_1, scale_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_7.run(buf91, primals_13, 32, grid=grid(32), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [scale_3], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 16, 1, 1), (16, 1, 16, 16))
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [scale_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_8.run(buf93, primals_15, 64, grid=grid(64), stream=stream0)
        del primals_15
        buf94 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [scale_4, input_7], Original ATen: [aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_mul_9.run(buf94, buf93, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf96 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf95, primals_17, primals_18, primals_19, primals_20, buf96, 16384, grid=grid(16384), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 72, 16, 16), (18432, 1, 1152, 72))
        buf98 = empty_strided_cuda((4, 72, 16, 16), (18432, 1, 1152, 72), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf97, primals_22, primals_23, primals_24, primals_25, buf98, 73728, grid=grid(73728), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf99, (4, 72, 8, 8), (4608, 1, 576, 72))
        buf100 = empty_strided_cuda((4, 72, 8, 8), (4608, 1, 576, 72), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf99, primals_27, primals_28, primals_29, primals_30, buf100, 18432, grid=grid(18432), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 24, 8, 8), (1536, 1, 192, 24))
        buf102 = empty_strided_cuda((4, 24, 8, 8), (1536, 1, 192, 24), torch.float32)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf101, primals_32, primals_33, primals_34, primals_35, buf102, 6144, grid=grid(6144), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 88, 8, 8), (5632, 1, 704, 88))
        buf104 = empty_strided_cuda((4, 88, 8, 8), (5632, 1, 704, 88), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf103, primals_37, primals_38, primals_39, primals_40, buf104, 22528, grid=grid(22528), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=88, bias=None)
        assert_size_stride(buf105, (4, 88, 8, 8), (5632, 1, 704, 88))
        buf106 = empty_strided_cuda((4, 88, 8, 8), (5632, 1, 704, 88), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf105, primals_42, primals_43, primals_44, primals_45, buf106, 22528, grid=grid(22528), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 24, 8, 8), (1536, 1, 192, 24))
        buf108 = empty_strided_cuda((4, 24, 8, 8), (1536, 1, 192, 24), torch.float32)
        # Topologically Sorted Source Nodes: [input_25, result], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf107, primals_47, primals_48, primals_49, primals_50, buf102, buf108, 6144, grid=grid(6144), stream=stream0)
        del primals_50
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf110 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_16.run(buf111, buf109, primals_52, primals_53, primals_54, primals_55, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_56, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf112, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf113 = empty_strided_cuda((4, 96, 4, 4), (1536, 1, 384, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_17.run(buf112, primals_57, primals_58, primals_59, primals_60, buf113, 6144, grid=grid(6144), stream=stream0)
        buf114 = empty_strided_cuda((4, 96, 1, 1), (96, 1, 384, 384), torch.float32)
        buf115 = reinterpret_tensor(buf114, (4, 96, 1, 1), (96, 1, 96, 96), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [input_31, scale_5], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_hardswish_mean_18.run(buf115, buf113, 384, 16, grid=grid(384), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_6], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 24, 1, 1), (24, 1, 24, 24))
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [scale_6, scale_7], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_19.run(buf117, primals_62, 96, grid=grid(96), stream=stream0)
        del primals_62
        # Topologically Sorted Source Nodes: [scale_8], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_63, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 96, 1, 1), (96, 1, 96, 96))
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [scale_8], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf119, primals_64, 384, grid=grid(384), stream=stream0)
        del primals_64
        buf120 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [input_31, scale_9, input_32], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_21.run(buf120, buf119, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_65, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 40, 4, 4), (640, 1, 160, 40))
        buf122 = empty_strided_cuda((4, 40, 4, 4), (640, 1, 160, 40), torch.float32)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf121, primals_66, primals_67, primals_68, primals_69, buf122, 2560, grid=grid(2560), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 240, 4, 4), (3840, 1, 960, 240))
        buf124 = empty_strided_cuda((4, 240, 4, 4), (3840, 1, 960, 240), torch.float32)
        buf125 = buf124; del buf124  # reuse
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23.run(buf125, buf123, primals_71, primals_72, primals_73, primals_74, 15360, grid=grid(15360), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_75, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf126, (4, 240, 4, 4), (3840, 1, 960, 240))
        buf127 = empty_strided_cuda((4, 240, 4, 4), (3840, 1, 960, 240), torch.float32)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf126, primals_76, primals_77, primals_78, primals_79, buf127, 15360, grid=grid(15360), stream=stream0)
        buf128 = empty_strided_cuda((4, 240, 1, 1), (240, 1, 960, 960), torch.float32)
        buf129 = reinterpret_tensor(buf128, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [input_40, scale_10], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_hardswish_mean_25.run(buf129, buf127, 960, 16, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_11], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 64, 1, 1), (64, 1, 64, 64))
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [scale_11, scale_12], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_26.run(buf131, primals_81, 256, grid=grid(256), stream=stream0)
        del primals_81
        # Topologically Sorted Source Nodes: [scale_13], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 240, 1, 1), (240, 1, 240, 240))
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [scale_13], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_27.run(buf133, primals_83, 960, grid=grid(960), stream=stream0)
        del primals_83
        buf134 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [input_40, scale_14, input_41], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_28.run(buf134, buf133, 15360, grid=grid(15360), stream=stream0)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_84, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 40, 4, 4), (640, 1, 160, 40))
        buf136 = empty_strided_cuda((4, 40, 4, 4), (640, 1, 160, 40), torch.float32)
        # Topologically Sorted Source Nodes: [input_43, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf135, primals_85, primals_86, primals_87, primals_88, buf122, buf136, 2560, grid=grid(2560), stream=stream0)
        del primals_88
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_89, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 240, 4, 4), (3840, 1, 960, 240))
        buf138 = empty_strided_cuda((4, 240, 4, 4), (3840, 1, 960, 240), torch.float32)
        buf139 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_23.run(buf139, buf137, primals_90, primals_91, primals_92, primals_93, 15360, grid=grid(15360), stream=stream0)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_94, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf140, (4, 240, 4, 4), (3840, 1, 960, 240))
        buf141 = empty_strided_cuda((4, 240, 4, 4), (3840, 1, 960, 240), torch.float32)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf140, primals_95, primals_96, primals_97, primals_98, buf141, 15360, grid=grid(15360), stream=stream0)
        buf142 = empty_strided_cuda((4, 240, 1, 1), (240, 1, 960, 960), torch.float32)
        buf143 = reinterpret_tensor(buf142, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [input_49, scale_15], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_hardswish_mean_25.run(buf143, buf141, 960, 16, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_16], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (4, 64, 1, 1), (64, 1, 64, 64))
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [scale_16, scale_17], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_26.run(buf145, primals_100, 256, grid=grid(256), stream=stream0)
        del primals_100
        # Topologically Sorted Source Nodes: [scale_18], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 240, 1, 1), (240, 1, 240, 240))
        buf147 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [scale_18], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_27.run(buf147, primals_102, 960, grid=grid(960), stream=stream0)
        del primals_102
        buf148 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [input_49, scale_19, input_50], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_28.run(buf148, buf147, 15360, grid=grid(15360), stream=stream0)
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 40, 4, 4), (640, 1, 160, 40))
        buf150 = empty_strided_cuda((4, 40, 4, 4), (640, 1, 160, 40), torch.float32)
        # Topologically Sorted Source Nodes: [input_52, result_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf149, primals_104, primals_105, primals_106, primals_107, buf136, buf150, 2560, grid=grid(2560), stream=stream0)
        del primals_107
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_108, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 120, 4, 4), (1920, 1, 480, 120))
        buf152 = empty_strided_cuda((4, 120, 4, 4), (1920, 1, 480, 120), torch.float32)
        buf153 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30.run(buf153, buf151, primals_109, primals_110, primals_111, primals_112, 7680, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_113, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf154, (4, 120, 4, 4), (1920, 1, 480, 120))
        buf155 = empty_strided_cuda((4, 120, 4, 4), (1920, 1, 480, 120), torch.float32)
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf154, primals_114, primals_115, primals_116, primals_117, buf155, 7680, grid=grid(7680), stream=stream0)
        buf156 = empty_strided_cuda((4, 120, 1, 1), (120, 1, 480, 480), torch.float32)
        buf157 = reinterpret_tensor(buf156, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [input_58, scale_20], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_hardswish_mean_32.run(buf157, buf155, 480, 16, grid=grid(480), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_21], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 32, 1, 1), (32, 1, 32, 32))
        buf159 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [scale_21, scale_22], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_33.run(buf159, primals_119, 128, grid=grid(128), stream=stream0)
        del primals_119
        # Topologically Sorted Source Nodes: [scale_23], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 120, 1, 1), (120, 1, 120, 120))
        buf161 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [scale_23], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_34.run(buf161, primals_121, 480, grid=grid(480), stream=stream0)
        del primals_121
        buf162 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [input_58, scale_24, input_59], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_35.run(buf162, buf161, 7680, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 48, 4, 4), (768, 1, 192, 48))
        buf164 = empty_strided_cuda((4, 48, 4, 4), (768, 1, 192, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf163, primals_123, primals_124, primals_125, primals_126, buf164, 3072, grid=grid(3072), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 144, 4, 4), (2304, 1, 576, 144))
        buf166 = empty_strided_cuda((4, 144, 4, 4), (2304, 1, 576, 144), torch.float32)
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [input_63, input_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_37.run(buf167, buf165, primals_128, primals_129, primals_130, primals_131, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_132, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf168, (4, 144, 4, 4), (2304, 1, 576, 144))
        buf169 = empty_strided_cuda((4, 144, 4, 4), (2304, 1, 576, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf168, primals_133, primals_134, primals_135, primals_136, buf169, 9216, grid=grid(9216), stream=stream0)
        buf170 = empty_strided_cuda((4, 144, 1, 1), (144, 1, 576, 576), torch.float32)
        buf171 = reinterpret_tensor(buf170, (4, 144, 1, 1), (144, 1, 144, 144), 0); del buf170  # reuse
        # Topologically Sorted Source Nodes: [input_67, scale_25], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_hardswish_mean_39.run(buf171, buf169, 576, 16, grid=grid(576), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_26], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 40, 1, 1), (40, 1, 40, 40))
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [scale_26, scale_27], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_40.run(buf173, primals_138, 160, grid=grid(160), stream=stream0)
        del primals_138
        # Topologically Sorted Source Nodes: [scale_28], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 144, 1, 1), (144, 1, 144, 144))
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [scale_28], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_41.run(buf175, primals_140, 576, grid=grid(576), stream=stream0)
        del primals_140
        buf176 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [input_67, scale_29, input_68], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_42.run(buf176, buf175, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_141, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 48, 4, 4), (768, 1, 192, 48))
        buf178 = empty_strided_cuda((4, 48, 4, 4), (768, 1, 192, 48), torch.float32)
        # Topologically Sorted Source Nodes: [input_70, result_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_43.run(buf177, primals_142, primals_143, primals_144, primals_145, buf164, buf178, 3072, grid=grid(3072), stream=stream0)
        del primals_145
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 288, 4, 4), (4608, 1, 1152, 288))
        buf180 = empty_strided_cuda((4, 288, 4, 4), (4608, 1, 1152, 288), torch.float32)
        buf181 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [input_72, input_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_44.run(buf181, buf179, primals_147, primals_148, primals_149, primals_150, 18432, grid=grid(18432), stream=stream0)
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_151, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=288, bias=None)
        assert_size_stride(buf182, (4, 288, 2, 2), (1152, 1, 576, 288))
        buf183 = empty_strided_cuda((4, 288, 2, 2), (1152, 1, 576, 288), torch.float32)
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_45.run(buf182, primals_152, primals_153, primals_154, primals_155, buf183, 4608, grid=grid(4608), stream=stream0)
        buf184 = empty_strided_cuda((4, 288, 1, 1), (288, 1, 288, 288), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, scale_30], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardswish_mean_46.run(buf183, buf184, 1152, grid=grid(1152), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_31], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 72, 1, 1), (72, 1, 72, 72))
        buf186 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [scale_31, scale_32], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_47.run(buf186, primals_157, 288, grid=grid(288), stream=stream0)
        del primals_157
        # Topologically Sorted Source Nodes: [scale_33], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 288, 1, 1), (288, 1, 288, 288))
        buf188 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [scale_33], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_48.run(buf188, primals_159, 1152, grid=grid(1152), stream=stream0)
        del primals_159
        buf189 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [input_76, scale_34, input_77], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_49.run(buf189, buf188, 4608, grid=grid(4608), stream=stream0)
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 96, 2, 2), (384, 1, 192, 96))
        buf191 = empty_strided_cuda((4, 96, 2, 2), (384, 1, 192, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf190, primals_161, primals_162, primals_163, primals_164, buf191, 1536, grid=grid(1536), stream=stream0)
        del primals_164
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, primals_165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (4, 576, 2, 2), (2304, 1, 1152, 576))
        buf193 = empty_strided_cuda((4, 576, 2, 2), (2304, 1, 1152, 576), torch.float32)
        buf194 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [input_81, input_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_51.run(buf194, buf192, primals_166, primals_167, primals_168, primals_169, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_170, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf195, (4, 576, 2, 2), (2304, 1, 1152, 576))
        buf196 = empty_strided_cuda((4, 576, 2, 2), (2304, 1, 1152, 576), torch.float32)
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf195, primals_171, primals_172, primals_173, primals_174, buf196, 9216, grid=grid(9216), stream=stream0)
        buf197 = empty_strided_cuda((4, 576, 1, 1), (576, 1, 576, 576), torch.float32)
        # Topologically Sorted Source Nodes: [input_85, scale_35], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardswish_mean_53.run(buf196, buf197, 2304, grid=grid(2304), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_36], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 144, 1, 1), (144, 1, 144, 144))
        buf199 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [scale_36, scale_37], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_54.run(buf199, primals_176, 576, grid=grid(576), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [scale_38], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 576, 1, 1), (576, 1, 576, 576))
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [scale_38], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf201, primals_178, 2304, grid=grid(2304), stream=stream0)
        del primals_178
        buf202 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_85, scale_39, input_86], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_56.run(buf202, buf201, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 96, 2, 2), (384, 1, 192, 96))
        buf204 = empty_strided_cuda((4, 96, 2, 2), (384, 1, 192, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_88, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf203, primals_180, primals_181, primals_182, primals_183, buf191, buf204, 1536, grid=grid(1536), stream=stream0)
        del primals_183
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 576, 2, 2), (2304, 1, 1152, 576))
        buf206 = empty_strided_cuda((4, 576, 2, 2), (2304, 1, 1152, 576), torch.float32)
        buf207 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [input_90, input_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_51.run(buf207, buf205, primals_185, primals_186, primals_187, primals_188, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_92], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_189, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf208, (4, 576, 2, 2), (2304, 1, 1152, 576))
        buf209 = empty_strided_cuda((4, 576, 2, 2), (2304, 1, 1152, 576), torch.float32)
        # Topologically Sorted Source Nodes: [input_93], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_52.run(buf208, primals_190, primals_191, primals_192, primals_193, buf209, 9216, grid=grid(9216), stream=stream0)
        buf210 = empty_strided_cuda((4, 576, 1, 1), (576, 1, 576, 576), torch.float32)
        # Topologically Sorted Source Nodes: [input_94, scale_40], Original ATen: [aten.hardswish, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardswish_mean_53.run(buf209, buf210, 2304, grid=grid(2304), stream=stream0)
        # Topologically Sorted Source Nodes: [scale_41], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 144, 1, 1), (144, 1, 144, 144))
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [scale_41, scale_42], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_54.run(buf212, primals_195, 576, grid=grid(576), stream=stream0)
        del primals_195
        # Topologically Sorted Source Nodes: [scale_43], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 576, 1, 1), (576, 1, 576, 576))
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [scale_43], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_55.run(buf214, primals_197, 2304, grid=grid(2304), stream=stream0)
        del primals_197
        buf215 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [input_94, scale_44, input_95], Original ATen: [aten.hardswish, aten.hardsigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_hardsigmoid_hardswish_mul_56.run(buf215, buf214, 9216, grid=grid(9216), stream=stream0)
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 96, 2, 2), (384, 1, 192, 96))
        buf217 = empty_strided_cuda((4, 96, 2, 2), (384, 1, 192, 96), torch.float32)
        # Topologically Sorted Source Nodes: [input_97, result_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf216, primals_199, primals_200, primals_201, primals_202, buf204, buf217, 1536, grid=grid(1536), stream=stream0)
        del primals_202
        # Topologically Sorted Source Nodes: [c3], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [c3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_58.run(buf219, primals_204, 8192, grid=grid(8192), stream=stream0)
        del primals_204
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 512, 1, 1), (512, 1, 512, 512))
        buf221 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf222 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [input_98, input_99], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf222, primals_206, buf221, 2048, grid=grid(2048), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 512, 1, 1), (512, 1, 512, 512))
        buf224 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf225 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [input_100, input_101], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf225, primals_208, buf224, 2048, grid=grid(2048), stream=stream0)
        del primals_208
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 512, 1, 1), (512, 1, 512, 512))
        buf227 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf228 = reinterpret_tensor(buf226, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf226  # reuse
        # Topologically Sorted Source Nodes: [input_102, input_103], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf228, primals_210, buf227, 2048, grid=grid(2048), stream=stream0)
        del primals_210
        buf229 = empty_strided_cuda((512, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_9], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_211, buf229, 262144, grid=grid(262144), stream=stream0)
        del primals_211
        buf230 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul_10], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_61.run(primals_212, buf230, 512, grid=grid(512), stream=stream0)
        del primals_212
        buf231 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_10, out], Original ATen: [aten.mul, aten.addmm]
        extern_kernels.addmm(buf230, reinterpret_tensor(buf228, (4, 512), (512, 1), 0), reinterpret_tensor(buf229, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf231)
        del buf230
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf219, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 512, 1, 1), (512, 1, 512, 512))
        buf233 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf234 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf234, primals_214, buf233, 2048, grid=grid(2048), stream=stream0)
        del primals_214
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 512, 1, 1), (512, 1, 512, 512))
        buf236 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf237 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [input_106, input_107], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf237, primals_216, buf236, 2048, grid=grid(2048), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, buf7, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 512, 1, 1), (512, 1, 512, 512))
        buf239 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf240 = reinterpret_tensor(buf238, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf238  # reuse
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf240, primals_218, buf239, 2048, grid=grid(2048), stream=stream0)
        del primals_218
        buf241 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_11, out_1], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_219, buf241, 262144, grid=grid(262144), stream=stream0)
        del primals_219
        buf242 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf240, (4, 512), (512, 1), 0), buf241, out=buf242)
        # Topologically Sorted Source Nodes: [input_110], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf219, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 512, 1, 1), (512, 1, 512, 512))
        buf244 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf245 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf245, primals_222, buf244, 2048, grid=grid(2048), stream=stream0)
        del primals_222
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 512, 1, 1), (512, 1, 512, 512))
        buf247 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf248 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [input_112, input_113], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf248, primals_224, buf247, 2048, grid=grid(2048), stream=stream0)
        del primals_224
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, buf10, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (4, 512, 1, 1), (512, 1, 512, 512))
        buf250 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf251 = reinterpret_tensor(buf249, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [input_114, input_115], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf251, primals_226, buf250, 2048, grid=grid(2048), stream=stream0)
        del primals_226
        buf252 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_13, out_2], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_227, buf252, 262144, grid=grid(262144), stream=stream0)
        del primals_227
        buf253 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf251, (4, 512), (512, 1), 0), buf252, out=buf253)
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf178, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf256 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_62.run(buf256, 4, grid=grid(4), stream=stream0)
        buf257 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_63.run(buf257, 4, grid=grid(4), stream=stream0)
        buf258 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_62.run(buf258, 4, grid=grid(4), stream=stream0)
        buf259 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_63.run(buf259, 4, grid=grid(4), stream=stream0)
        buf260 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_64.run(buf260, 4, grid=grid(4), stream=stream0)
        buf262 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_64.run(buf262, 4, grid=grid(4), stream=stream0)
        buf263 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [conv2d_61, interpolate, p2], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sub_65.run(buf263, buf256, buf258, buf219, buf259, buf260, buf257, buf262, primals_230, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_230
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, buf11, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf265 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf266 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [input_116, input_117], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf266, primals_232, buf265, 8192, grid=grid(8192), stream=stream0)
        del primals_232
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 512, 1, 1), (512, 1, 512, 512))
        buf268 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf269 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [input_118, input_119], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf269, primals_234, buf268, 2048, grid=grid(2048), stream=stream0)
        del primals_234
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, buf13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 512, 1, 1), (512, 1, 512, 512))
        buf271 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf272 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [input_120, input_121], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf272, primals_236, buf271, 2048, grid=grid(2048), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 512, 1, 1), (512, 1, 512, 512))
        buf274 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf275 = reinterpret_tensor(buf273, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [input_122, input_123], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf275, primals_238, buf274, 2048, grid=grid(2048), stream=stream0)
        del primals_238
        buf276 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_15, out_3], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_239, buf276, 262144, grid=grid(262144), stream=stream0)
        del primals_239
        buf277 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf275, (4, 512), (512, 1), 0), buf276, out=buf277)
        buf254 = buf253; del buf253  # reuse
        buf278 = buf277; del buf277  # reuse
        # Topologically Sorted Source Nodes: [mul_14, out_2, iadd_7, mul_16, out_3, iadd_8], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_67.run(buf254, buf278, buf231, buf242, primals_220, primals_228, primals_240, 2048, grid=grid(2048), stream=stream0)
        del primals_228
        del primals_240
        buf279 = empty_strided_cuda((4, 18, 512), (9216, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_16, out_3, iadd_8], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_68.run(buf278, buf254, buf231, buf242, primals_220, buf279, 36864, grid=grid(36864), stream=stream0)
        del buf231
        del buf242
        del buf254
        del primals_220
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf263, buf15, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf281 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf282 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [input_124, input_125], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf282, primals_242, buf281, 8192, grid=grid(8192), stream=stream0)
        del primals_242
        # Topologically Sorted Source Nodes: [input_126], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, buf16, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 512, 1, 1), (512, 1, 512, 512))
        buf284 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf285 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [input_126, input_127], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf285, primals_244, buf284, 2048, grid=grid(2048), stream=stream0)
        del primals_244
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, buf17, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 512, 1, 1), (512, 1, 512, 512))
        buf287 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf288 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf288, primals_246, buf287, 2048, grid=grid(2048), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, buf18, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 512, 1, 1), (512, 1, 512, 512))
        buf290 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf291 = reinterpret_tensor(buf289, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [input_130, input_131], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf291, primals_248, buf290, 2048, grid=grid(2048), stream=stream0)
        del primals_248
        buf292 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_17, out_4], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_249, buf292, 262144, grid=grid(262144), stream=stream0)
        del primals_249
        buf293 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf291, (4, 512), (512, 1), 0), buf292, out=buf293)
        buf294 = empty_strided_cuda((4, 18, 512), (9216, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_18, out_4, iadd_9], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_69.run(buf279, buf293, primals_250, buf294, 36864, grid=grid(36864), stream=stream0)
        del primals_250
        # Topologically Sorted Source Nodes: [input_132], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf263, buf19, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf296 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf297 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [input_132, input_133], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf297, primals_252, buf296, 8192, grid=grid(8192), stream=stream0)
        del primals_252
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, buf20, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 512, 1, 1), (512, 1, 512, 512))
        buf299 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf300 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [input_134, input_135], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf300, primals_254, buf299, 2048, grid=grid(2048), stream=stream0)
        del primals_254
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, buf21, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 512, 1, 1), (512, 1, 512, 512))
        buf302 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf303 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [input_136, input_137], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf303, primals_256, buf302, 2048, grid=grid(2048), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [input_138], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, buf22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 512, 1, 1), (512, 1, 512, 512))
        buf305 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf306 = reinterpret_tensor(buf304, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [input_138, input_139], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf306, primals_258, buf305, 2048, grid=grid(2048), stream=stream0)
        del primals_258
        buf307 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_19, out_5], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_259, buf307, 262144, grid=grid(262144), stream=stream0)
        del primals_259
        buf308 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf306, (4, 512), (512, 1), 0), buf307, out=buf308)
        buf309 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [mul_20, out_5, iadd_10], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_70.run(buf294, buf308, primals_260, buf309, 36864, grid=grid(36864), stream=stream0)
        del primals_260
        # Topologically Sorted Source Nodes: [input_140], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf263, buf23, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf311 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf312 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [input_140, input_141], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf312, primals_262, buf311, 8192, grid=grid(8192), stream=stream0)
        del primals_262
        # Topologically Sorted Source Nodes: [input_142], Original ATen: [aten.convolution]
        buf313 = extern_kernels.convolution(buf312, buf24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (4, 512, 1, 1), (512, 1, 512, 512))
        buf314 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf315 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [input_142, input_143], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf315, primals_264, buf314, 2048, grid=grid(2048), stream=stream0)
        del primals_264
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf315, buf25, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 512, 1, 1), (512, 1, 512, 512))
        buf317 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf318 = buf316; del buf316  # reuse
        # Topologically Sorted Source Nodes: [input_144, input_145], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf318, primals_266, buf317, 2048, grid=grid(2048), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [input_146], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, buf26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 512, 1, 1), (512, 1, 512, 512))
        buf320 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf321 = reinterpret_tensor(buf319, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [input_146, input_147], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf321, primals_268, buf320, 2048, grid=grid(2048), stream=stream0)
        del primals_268
        buf322 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_21, out_6], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_269, buf322, 262144, grid=grid(262144), stream=stream0)
        del primals_269
        buf323 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf321, (4, 512), (512, 1), 0), buf322, out=buf323)
        buf324 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [mul_22, out_6, iadd_11], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_71.run(buf309, buf323, primals_270, buf324, 36864, grid=grid(36864), stream=stream0)
        del primals_270
        # Topologically Sorted Source Nodes: [conv2d_78], Original ATen: [aten.convolution]
        buf325 = extern_kernels.convolution(buf108, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf325, (4, 512, 8, 8), (32768, 1, 4096, 512))
        buf326 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf326, 8, grid=grid(8), stream=stream0)
        buf327 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_73.run(buf327, 8, grid=grid(8), stream=stream0)
        buf328 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_72.run(buf328, 8, grid=grid(8), stream=stream0)
        buf329 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_73.run(buf329, 8, grid=grid(8), stream=stream0)
        buf330 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_74.run(buf330, 8, grid=grid(8), stream=stream0)
        buf332 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [interpolate_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_74.run(buf332, 8, grid=grid(8), stream=stream0)
        buf333 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [conv2d_78, interpolate_1, p1], Original ATen: [aten.convolution, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_mul_sub_75.run(buf333, buf326, buf328, buf263, buf329, buf330, buf327, buf332, primals_272, 2048, 64, grid=grid(2048, 64), stream=stream0)
        del primals_272
        # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, buf27, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf335 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf336 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [input_148, input_149], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf336, primals_274, buf335, 32768, grid=grid(32768), stream=stream0)
        del primals_274
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, buf28, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf338 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf339 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [input_150, input_151], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf339, primals_276, buf338, 8192, grid=grid(8192), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [input_152], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, buf29, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 512, 1, 1), (512, 1, 512, 512))
        buf341 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf342 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [input_152, input_153], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf342, primals_278, buf341, 2048, grid=grid(2048), stream=stream0)
        del primals_278
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, buf30, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 512, 1, 1), (512, 1, 512, 512))
        buf344 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf345 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [input_154, input_155], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf345, primals_280, buf344, 2048, grid=grid(2048), stream=stream0)
        del primals_280
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, buf31, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 512, 1, 1), (512, 1, 512, 512))
        buf347 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf348 = reinterpret_tensor(buf346, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [input_156, input_157], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf348, primals_282, buf347, 2048, grid=grid(2048), stream=stream0)
        del primals_282
        buf349 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_23, out_7], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_283, buf349, 262144, grid=grid(262144), stream=stream0)
        del primals_283
        buf350 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [out_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf348, (4, 512), (512, 1), 0), buf349, out=buf350)
        buf351 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [mul_24, out_7, iadd_12], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_77.run(buf324, buf350, primals_284, buf351, 36864, grid=grid(36864), stream=stream0)
        del primals_284
        # Topologically Sorted Source Nodes: [input_158], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf333, buf32, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf353 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf354 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [input_158, input_159], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf354, primals_286, buf353, 32768, grid=grid(32768), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf354, buf33, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf356 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf357 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [input_160, input_161], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf357, primals_288, buf356, 8192, grid=grid(8192), stream=stream0)
        del primals_288
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, buf34, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 512, 1, 1), (512, 1, 512, 512))
        buf359 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf360 = buf358; del buf358  # reuse
        # Topologically Sorted Source Nodes: [input_162, input_163], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf360, primals_290, buf359, 2048, grid=grid(2048), stream=stream0)
        del primals_290
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, buf35, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 512, 1, 1), (512, 1, 512, 512))
        buf362 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf363 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [input_164, input_165], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf363, primals_292, buf362, 2048, grid=grid(2048), stream=stream0)
        del primals_292
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf363, buf36, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 512, 1, 1), (512, 1, 512, 512))
        buf365 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf366 = reinterpret_tensor(buf364, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf364  # reuse
        # Topologically Sorted Source Nodes: [input_166, input_167], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf366, primals_294, buf365, 2048, grid=grid(2048), stream=stream0)
        del primals_294
        buf367 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_25, out_8], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_295, buf367, 262144, grid=grid(262144), stream=stream0)
        del primals_295
        buf368 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf366, (4, 512), (512, 1), 0), buf367, out=buf368)
        buf369 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [mul_26, out_8, iadd_13], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_78.run(buf351, buf368, primals_296, buf369, 36864, grid=grid(36864), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [input_168], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf333, buf37, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf371 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf372 = buf370; del buf370  # reuse
        # Topologically Sorted Source Nodes: [input_168, input_169], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf372, primals_298, buf371, 32768, grid=grid(32768), stream=stream0)
        del primals_298
        # Topologically Sorted Source Nodes: [input_170], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, buf38, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf374 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf375 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [input_170, input_171], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf375, primals_300, buf374, 8192, grid=grid(8192), stream=stream0)
        del primals_300
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, buf39, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 512, 1, 1), (512, 1, 512, 512))
        buf377 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf378 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [input_172, input_173], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf378, primals_302, buf377, 2048, grid=grid(2048), stream=stream0)
        del primals_302
        # Topologically Sorted Source Nodes: [input_174], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, buf40, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 512, 1, 1), (512, 1, 512, 512))
        buf380 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf381 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [input_174, input_175], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf381, primals_304, buf380, 2048, grid=grid(2048), stream=stream0)
        del primals_304
        # Topologically Sorted Source Nodes: [input_176], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, buf41, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf382, (4, 512, 1, 1), (512, 1, 512, 512))
        buf383 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf384 = reinterpret_tensor(buf382, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf382  # reuse
        # Topologically Sorted Source Nodes: [input_176, input_177], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf384, primals_306, buf383, 2048, grid=grid(2048), stream=stream0)
        del primals_306
        buf385 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_27, out_9], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_307, buf385, 262144, grid=grid(262144), stream=stream0)
        del primals_307
        buf386 = buf368; del buf368  # reuse
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf384, (4, 512), (512, 1), 0), buf385, out=buf386)
        buf387 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [mul_28, out_9, iadd_14], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_79.run(buf369, buf386, primals_308, buf387, 36864, grid=grid(36864), stream=stream0)
        del primals_308
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf333, buf42, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf389 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf390 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [input_178, input_179], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf390, primals_310, buf389, 32768, grid=grid(32768), stream=stream0)
        del primals_310
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        buf391 = extern_kernels.convolution(buf390, buf43, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf391, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf392 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf393 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [input_180, input_181], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf393, primals_312, buf392, 8192, grid=grid(8192), stream=stream0)
        del primals_312
        # Topologically Sorted Source Nodes: [input_182], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, buf44, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 512, 1, 1), (512, 1, 512, 512))
        buf395 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf396 = buf394; del buf394  # reuse
        # Topologically Sorted Source Nodes: [input_182, input_183], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf396, primals_314, buf395, 2048, grid=grid(2048), stream=stream0)
        del primals_314
        # Topologically Sorted Source Nodes: [input_184], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, buf45, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 512, 1, 1), (512, 1, 512, 512))
        buf398 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf399 = buf397; del buf397  # reuse
        # Topologically Sorted Source Nodes: [input_184, input_185], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf399, primals_316, buf398, 2048, grid=grid(2048), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [input_186], Original ATen: [aten.convolution]
        buf400 = extern_kernels.convolution(buf399, buf46, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf400, (4, 512, 1, 1), (512, 1, 512, 512))
        buf401 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf402 = reinterpret_tensor(buf400, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [input_186, input_187], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf402, primals_318, buf401, 2048, grid=grid(2048), stream=stream0)
        del primals_318
        buf403 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_29, out_10], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_319, buf403, 262144, grid=grid(262144), stream=stream0)
        del primals_319
        buf404 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf402, (4, 512), (512, 1), 0), buf403, out=buf404)
        buf405 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [mul_30, out_10, iadd_15], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_80.run(buf387, buf404, primals_320, buf405, 36864, grid=grid(36864), stream=stream0)
        del primals_320
        # Topologically Sorted Source Nodes: [input_188], Original ATen: [aten.convolution]
        buf406 = extern_kernels.convolution(buf333, buf47, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf407 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf408 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [input_188, input_189], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf408, primals_322, buf407, 32768, grid=grid(32768), stream=stream0)
        del primals_322
        # Topologically Sorted Source Nodes: [input_190], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, buf48, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf410 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf411 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [input_190, input_191], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf411, primals_324, buf410, 8192, grid=grid(8192), stream=stream0)
        del primals_324
        # Topologically Sorted Source Nodes: [input_192], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, buf49, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 512, 1, 1), (512, 1, 512, 512))
        buf413 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf414 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [input_192, input_193], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf414, primals_326, buf413, 2048, grid=grid(2048), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf414, buf50, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (4, 512, 1, 1), (512, 1, 512, 512))
        buf416 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf417 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [input_194, input_195], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf417, primals_328, buf416, 2048, grid=grid(2048), stream=stream0)
        del primals_328
        # Topologically Sorted Source Nodes: [input_196], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, buf51, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (4, 512, 1, 1), (512, 1, 512, 512))
        buf419 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf420 = reinterpret_tensor(buf418, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf418  # reuse
        # Topologically Sorted Source Nodes: [input_196, input_197], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf420, primals_330, buf419, 2048, grid=grid(2048), stream=stream0)
        del primals_330
        buf421 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_31, out_11], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_331, buf421, 262144, grid=grid(262144), stream=stream0)
        del primals_331
        buf422 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf420, (4, 512), (512, 1), 0), buf421, out=buf422)
        buf423 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [mul_32, out_11, iadd_16], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_81.run(buf405, buf422, primals_332, buf423, 36864, grid=grid(36864), stream=stream0)
        del primals_332
        # Topologically Sorted Source Nodes: [input_198], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf333, buf52, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf425 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf426 = buf424; del buf424  # reuse
        # Topologically Sorted Source Nodes: [input_198, input_199], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf426, primals_334, buf425, 32768, grid=grid(32768), stream=stream0)
        del primals_334
        # Topologically Sorted Source Nodes: [input_200], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, buf53, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf428 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf429 = buf427; del buf427  # reuse
        # Topologically Sorted Source Nodes: [input_200, input_201], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf429, primals_336, buf428, 8192, grid=grid(8192), stream=stream0)
        del primals_336
        # Topologically Sorted Source Nodes: [input_202], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, buf54, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (4, 512, 1, 1), (512, 1, 512, 512))
        buf431 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf432 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [input_202, input_203], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf432, primals_338, buf431, 2048, grid=grid(2048), stream=stream0)
        del primals_338
        # Topologically Sorted Source Nodes: [input_204], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, buf55, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (4, 512, 1, 1), (512, 1, 512, 512))
        buf434 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf435 = buf433; del buf433  # reuse
        # Topologically Sorted Source Nodes: [input_204, input_205], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf435, primals_340, buf434, 2048, grid=grid(2048), stream=stream0)
        del primals_340
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, buf56, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (4, 512, 1, 1), (512, 1, 512, 512))
        buf437 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf438 = reinterpret_tensor(buf436, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf436  # reuse
        # Topologically Sorted Source Nodes: [input_206, input_207], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf438, primals_342, buf437, 2048, grid=grid(2048), stream=stream0)
        del primals_342
        buf439 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_33, out_12], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_343, buf439, 262144, grid=grid(262144), stream=stream0)
        del primals_343
        buf440 = buf422; del buf422  # reuse
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf438, (4, 512), (512, 1), 0), buf439, out=buf440)
        buf441 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [mul_34, out_12, iadd_17], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_82.run(buf423, buf440, primals_344, buf441, 36864, grid=grid(36864), stream=stream0)
        del primals_344
        # Topologically Sorted Source Nodes: [input_208], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf333, buf57, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf443 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf444 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [input_208, input_209], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf444, primals_346, buf443, 32768, grid=grid(32768), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [input_210], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, buf58, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf446 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf447 = buf445; del buf445  # reuse
        # Topologically Sorted Source Nodes: [input_210, input_211], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf447, primals_348, buf446, 8192, grid=grid(8192), stream=stream0)
        del primals_348
        # Topologically Sorted Source Nodes: [input_212], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, buf59, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 512, 1, 1), (512, 1, 512, 512))
        buf449 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf450 = buf448; del buf448  # reuse
        # Topologically Sorted Source Nodes: [input_212, input_213], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf450, primals_350, buf449, 2048, grid=grid(2048), stream=stream0)
        del primals_350
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, buf60, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (4, 512, 1, 1), (512, 1, 512, 512))
        buf452 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf453 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [input_214, input_215], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf453, primals_352, buf452, 2048, grid=grid(2048), stream=stream0)
        del primals_352
        # Topologically Sorted Source Nodes: [input_216], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, buf61, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf454, (4, 512, 1, 1), (512, 1, 512, 512))
        buf455 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf456 = reinterpret_tensor(buf454, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf454  # reuse
        # Topologically Sorted Source Nodes: [input_216, input_217], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf456, primals_354, buf455, 2048, grid=grid(2048), stream=stream0)
        del primals_354
        buf457 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_35, out_13], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_355, buf457, 262144, grid=grid(262144), stream=stream0)
        del primals_355
        buf458 = buf440; del buf440  # reuse
        # Topologically Sorted Source Nodes: [out_13], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf456, (4, 512), (512, 1), 0), buf457, out=buf458)
        buf459 = buf423; del buf423  # reuse
        # Topologically Sorted Source Nodes: [mul_36, out_13, iadd_18], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_83.run(buf441, buf458, primals_356, buf459, 36864, grid=grid(36864), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [input_218], Original ATen: [aten.convolution]
        buf460 = extern_kernels.convolution(buf333, buf62, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf460, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf461 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf462 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [input_218, input_219], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf462, primals_358, buf461, 32768, grid=grid(32768), stream=stream0)
        del primals_358
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, buf63, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf464 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf465 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [input_220, input_221], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf465, primals_360, buf464, 8192, grid=grid(8192), stream=stream0)
        del primals_360
        # Topologically Sorted Source Nodes: [input_222], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, buf64, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf466, (4, 512, 1, 1), (512, 1, 512, 512))
        buf467 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf468 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [input_222, input_223], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf468, primals_362, buf467, 2048, grid=grid(2048), stream=stream0)
        del primals_362
        # Topologically Sorted Source Nodes: [input_224], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf468, buf65, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf469, (4, 512, 1, 1), (512, 1, 512, 512))
        buf470 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf471 = buf469; del buf469  # reuse
        # Topologically Sorted Source Nodes: [input_224, input_225], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf471, primals_364, buf470, 2048, grid=grid(2048), stream=stream0)
        del primals_364
        # Topologically Sorted Source Nodes: [input_226], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, buf66, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (4, 512, 1, 1), (512, 1, 512, 512))
        buf473 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf474 = reinterpret_tensor(buf472, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf472  # reuse
        # Topologically Sorted Source Nodes: [input_226, input_227], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf474, primals_366, buf473, 2048, grid=grid(2048), stream=stream0)
        del primals_366
        buf475 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_37, out_14], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_367, buf475, 262144, grid=grid(262144), stream=stream0)
        del primals_367
        buf476 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf474, (4, 512), (512, 1), 0), buf475, out=buf476)
        buf477 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [mul_38, out_14, iadd_19], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_84.run(buf459, buf476, primals_368, buf477, 36864, grid=grid(36864), stream=stream0)
        del primals_368
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.convolution]
        buf478 = extern_kernels.convolution(buf333, buf67, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf478, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf479 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf480 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [input_228, input_229], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf480, primals_370, buf479, 32768, grid=grid(32768), stream=stream0)
        del primals_370
        # Topologically Sorted Source Nodes: [input_230], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, buf68, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf482 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf483 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [input_230, input_231], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf483, primals_372, buf482, 8192, grid=grid(8192), stream=stream0)
        del primals_372
        # Topologically Sorted Source Nodes: [input_232], Original ATen: [aten.convolution]
        buf484 = extern_kernels.convolution(buf483, buf69, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf484, (4, 512, 1, 1), (512, 1, 512, 512))
        buf485 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf486 = buf484; del buf484  # reuse
        # Topologically Sorted Source Nodes: [input_232, input_233], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf486, primals_374, buf485, 2048, grid=grid(2048), stream=stream0)
        del primals_374
        # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, buf70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (4, 512, 1, 1), (512, 1, 512, 512))
        buf488 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf489 = buf487; del buf487  # reuse
        # Topologically Sorted Source Nodes: [input_234, input_235], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf489, primals_376, buf488, 2048, grid=grid(2048), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [input_236], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, buf71, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 512, 1, 1), (512, 1, 512, 512))
        buf491 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf492 = reinterpret_tensor(buf490, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [input_236, input_237], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf492, primals_378, buf491, 2048, grid=grid(2048), stream=stream0)
        del primals_378
        buf493 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_39, out_15], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_379, buf493, 262144, grid=grid(262144), stream=stream0)
        del primals_379
        buf494 = buf476; del buf476  # reuse
        # Topologically Sorted Source Nodes: [out_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf492, (4, 512), (512, 1), 0), buf493, out=buf494)
        buf495 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [mul_40, out_15, iadd_20], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_85.run(buf477, buf494, primals_380, buf495, 36864, grid=grid(36864), stream=stream0)
        del primals_380
        # Topologically Sorted Source Nodes: [input_238], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf333, buf72, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf497 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf498 = buf496; del buf496  # reuse
        # Topologically Sorted Source Nodes: [input_238, input_239], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf498, primals_382, buf497, 32768, grid=grid(32768), stream=stream0)
        del primals_382
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, buf73, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf500 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf501 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [input_240, input_241], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf501, primals_384, buf500, 8192, grid=grid(8192), stream=stream0)
        del primals_384
        # Topologically Sorted Source Nodes: [input_242], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, buf74, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 512, 1, 1), (512, 1, 512, 512))
        buf503 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf504 = buf502; del buf502  # reuse
        # Topologically Sorted Source Nodes: [input_242, input_243], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf504, primals_386, buf503, 2048, grid=grid(2048), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [input_244], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, buf75, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf505, (4, 512, 1, 1), (512, 1, 512, 512))
        buf506 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf507 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [input_244, input_245], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf507, primals_388, buf506, 2048, grid=grid(2048), stream=stream0)
        del primals_388
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, buf76, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (4, 512, 1, 1), (512, 1, 512, 512))
        buf509 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf510 = reinterpret_tensor(buf508, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf508  # reuse
        # Topologically Sorted Source Nodes: [input_246, input_247], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf510, primals_390, buf509, 2048, grid=grid(2048), stream=stream0)
        del primals_390
        buf511 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_41, out_16], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_391, buf511, 262144, grid=grid(262144), stream=stream0)
        del primals_391
        buf512 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf510, (4, 512), (512, 1), 0), buf511, out=buf512)
        buf513 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [mul_42, out_16, iadd_21], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_86.run(buf495, buf512, primals_392, buf513, 36864, grid=grid(36864), stream=stream0)
        del primals_392
        # Topologically Sorted Source Nodes: [input_248], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf333, buf77, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf515 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.bool)
        buf516 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [input_248, input_249], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_76.run(buf516, primals_394, buf515, 32768, grid=grid(32768), stream=stream0)
        del primals_394
        # Topologically Sorted Source Nodes: [input_250], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, buf78, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf518 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.bool)
        buf519 = buf517; del buf517  # reuse
        # Topologically Sorted Source Nodes: [input_250, input_251], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_66.run(buf519, primals_396, buf518, 8192, grid=grid(8192), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [input_252], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, buf79, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (4, 512, 1, 1), (512, 1, 512, 512))
        buf521 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf522 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [input_252, input_253], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf522, primals_398, buf521, 2048, grid=grid(2048), stream=stream0)
        del primals_398
        # Topologically Sorted Source Nodes: [input_254], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, buf80, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (4, 512, 1, 1), (512, 1, 512, 512))
        buf524 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf525 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [input_254, input_255], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf525, primals_400, buf524, 2048, grid=grid(2048), stream=stream0)
        del primals_400
        # Topologically Sorted Source Nodes: [input_256], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, buf81, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 512, 1, 1), (512, 1, 512, 512))
        buf527 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 512, 512), torch.bool)
        buf528 = reinterpret_tensor(buf526, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf526  # reuse
        # Topologically Sorted Source Nodes: [input_256, input_257], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf528, primals_402, buf527, 2048, grid=grid(2048), stream=stream0)
        del primals_402
        buf529 = empty_strided_cuda((512, 512), (1, 512), torch.float32)
        # Topologically Sorted Source Nodes: [mul_43, out_17], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_60.run(primals_403, buf529, 262144, grid=grid(262144), stream=stream0)
        del primals_403
        buf530 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf528, (4, 512), (512, 1), 0), buf529, out=buf530)
        buf531 = buf495; del buf495  # reuse
        # Topologically Sorted Source Nodes: [mul_44, out_17, iadd_22], Original ATen: [aten.mul, aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_mul_87.run(buf513, buf530, primals_404, buf531, 36864, grid=grid(36864), stream=stream0)
        del buf530
        del primals_404
        buf532 = reinterpret_tensor(buf513, (4, 18, 512), (512, 2048, 1), 0); del buf513  # reuse
        # Topologically Sorted Source Nodes: [repeat_1, w_1], Original ATen: [aten.repeat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_repeat_88.run(buf531, primals_405, buf532, 36864, grid=grid(36864), stream=stream0)
        del buf531
        del primals_405
    return (buf532, buf0, buf1, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_19, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_49, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_63, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_84, primals_85, primals_86, primals_87, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_101, primals_103, primals_104, primals_105, primals_106, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_141, primals_142, primals_143, primals_144, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_158, primals_160, primals_161, primals_162, primals_163, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_179, primals_180, primals_181, primals_182, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_196, primals_198, primals_199, primals_200, primals_201, primals_203, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, primals_229, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, primals_271, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf84, buf85, buf89, buf91, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf111, buf112, buf115, buf117, buf119, buf120, buf121, buf122, buf123, buf125, buf126, buf129, buf131, buf133, buf134, buf135, buf136, buf137, buf139, buf140, buf143, buf145, buf147, buf148, buf149, buf150, buf151, buf153, buf154, buf157, buf159, buf161, buf162, buf163, buf164, buf165, buf167, buf168, buf171, buf173, buf175, buf176, buf177, buf178, buf179, buf181, buf182, buf184, buf186, buf188, buf189, buf190, buf191, buf192, buf194, buf195, buf197, buf199, buf201, buf202, buf203, buf204, buf205, buf207, buf208, buf210, buf212, buf214, buf215, buf216, buf217, buf219, buf221, buf222, buf224, buf225, buf227, reinterpret_tensor(buf228, (4, 512), (512, 1), 0), buf233, buf234, buf236, buf237, buf239, reinterpret_tensor(buf240, (4, 512), (512, 1), 0), buf244, buf245, buf247, buf248, buf250, reinterpret_tensor(buf251, (4, 512), (512, 1), 0), buf256, buf257, buf258, buf259, buf260, buf262, buf263, buf265, buf266, buf268, buf269, buf271, buf272, buf274, reinterpret_tensor(buf275, (4, 512), (512, 1), 0), buf281, buf282, buf284, buf285, buf287, buf288, buf290, reinterpret_tensor(buf291, (4, 512), (512, 1), 0), buf296, buf297, buf299, buf300, buf302, buf303, buf305, reinterpret_tensor(buf306, (4, 512), (512, 1), 0), buf311, buf312, buf314, buf315, buf317, buf318, buf320, reinterpret_tensor(buf321, (4, 512), (512, 1), 0), buf326, buf327, buf328, buf329, buf330, buf332, buf333, buf335, buf336, buf338, buf339, buf341, buf342, buf344, buf345, buf347, reinterpret_tensor(buf348, (4, 512), (512, 1), 0), buf353, buf354, buf356, buf357, buf359, buf360, buf362, buf363, buf365, reinterpret_tensor(buf366, (4, 512), (512, 1), 0), buf371, buf372, buf374, buf375, buf377, buf378, buf380, buf381, buf383, reinterpret_tensor(buf384, (4, 512), (512, 1), 0), buf389, buf390, buf392, buf393, buf395, buf396, buf398, buf399, buf401, reinterpret_tensor(buf402, (4, 512), (512, 1), 0), buf407, buf408, buf410, buf411, buf413, buf414, buf416, buf417, buf419, reinterpret_tensor(buf420, (4, 512), (512, 1), 0), buf425, buf426, buf428, buf429, buf431, buf432, buf434, buf435, buf437, reinterpret_tensor(buf438, (4, 512), (512, 1), 0), buf443, buf444, buf446, buf447, buf449, buf450, buf452, buf453, buf455, reinterpret_tensor(buf456, (4, 512), (512, 1), 0), buf461, buf462, buf464, buf465, buf467, buf468, buf470, buf471, buf473, reinterpret_tensor(buf474, (4, 512), (512, 1), 0), buf479, buf480, buf482, buf483, buf485, buf486, buf488, buf489, buf491, reinterpret_tensor(buf492, (4, 512), (512, 1), 0), buf497, buf498, buf500, buf501, buf503, buf504, buf506, buf507, buf509, reinterpret_tensor(buf510, (4, 512), (512, 1), 0), buf515, buf516, buf518, buf519, buf521, buf522, buf524, buf525, buf527, reinterpret_tensor(buf528, (4, 512), (512, 1), 0), reinterpret_tensor(buf529, (512, 512), (512, 1), 0), reinterpret_tensor(buf511, (512, 512), (512, 1), 0), reinterpret_tensor(buf493, (512, 512), (512, 1), 0), reinterpret_tensor(buf475, (512, 512), (512, 1), 0), reinterpret_tensor(buf457, (512, 512), (512, 1), 0), reinterpret_tensor(buf439, (512, 512), (512, 1), 0), reinterpret_tensor(buf421, (512, 512), (512, 1), 0), reinterpret_tensor(buf403, (512, 512), (512, 1), 0), reinterpret_tensor(buf385, (512, 512), (512, 1), 0), reinterpret_tensor(buf367, (512, 512), (512, 1), 0), reinterpret_tensor(buf349, (512, 512), (512, 1), 0), reinterpret_tensor(buf322, (512, 512), (512, 1), 0), reinterpret_tensor(buf307, (512, 512), (512, 1), 0), reinterpret_tensor(buf292, (512, 512), (512, 1), 0), reinterpret_tensor(buf276, (512, 512), (512, 1), 0), reinterpret_tensor(buf252, (512, 512), (512, 1), 0), reinterpret_tensor(buf241, (512, 512), (512, 1), 0), buf229, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((72, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((88, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((88, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((88, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((24, 88, 1, 1), (88, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((96, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((96, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((24, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((96, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((40, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((240, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((240, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((48, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((144, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((144, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((40, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((144, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((48, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((288, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((288, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((72, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((288, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((96, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((144, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((576, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((144, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((576, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((512, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((512, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((18, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
