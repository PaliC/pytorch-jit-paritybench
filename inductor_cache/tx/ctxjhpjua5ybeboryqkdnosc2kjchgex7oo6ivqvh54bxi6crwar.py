# AOT ID: ['39_forward']
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


# kernel path: inductor_cache/bo/cbol6kmkhdzwkntbee3wxnrbpetxgorlwntotzwbfw7hwycwiuyd.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_7 => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %convolution_5), kwargs = {})
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/cu/ccuccdbbuncr5solhgqhczrrc6uxhgqw7tjstzy4fd5bus3lqqpp.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_18 => add_2
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_11, %convolution_14), kwargs = {})
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/as/cas47k5weedjtuz6qwsiis3zp6tbeujigwanjxo6wvmdqpcizqqb.py
# Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_29 => add_4
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_20, %convolution_23), kwargs = {})
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/up/cup64xrd7g5ldlxld27btj45wsotgj4hyc77pvxp3wbelrafyidx.py
# Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   input_33 => add_5
#   input_34 => mean
# Graph fragment:
#   %add_5 : [num_users=6] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_26), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_5, [-1, -2], True), kwargs = {})
triton_per_fused_add_mean_3 = async_compile.triton('triton_per_fused_add_mean_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_3(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 256*x0), None)
    tmp1 = tl.load(in_out_ptr0 + (r1 + 256*x0), None)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tl.store(in_out_ptr0 + (r1 + 256*x0), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/yo/cyolgtjzeii3r5xeposlu7z37jrqsxkqjiqeht7pmdmdykw6aktt.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsampled_ppm_out => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/my/cmymncii7f4r4i2rn6rfcgrvwmphqq3t3tfms6dtodn3rzgcpof7.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out => add_7, clamp_max
# Graph fragment:
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_7, 0), kwargs = {})
triton_poi_fused_add_clamp_5 = async_compile.triton('triton_poi_fused_add_clamp_5', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 0, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yc/cyccdprkxflq5pkqdq3nmj7fzjwrepvlxoy4kk3vsw4sbkrdywsq.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out => add_6, clamp_max_2, clamp_min, clamp_min_2, convert_element_type, iota, mul, sub, sub_2
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_6 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.0625), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_6 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_6', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.0625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/r5/cr5t5hulyufaklhp5ns7p46yfzijlctje5htlwxexirqfhx5g73f.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsampled_ppm_out => _unsafe_index, _unsafe_index_1, add_10, mul_2, sub_3
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_27, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_27, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_7 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tmp9 - tmp9
    tmp16 = tmp14 * tmp15
    tmp17 = tmp9 + tmp16
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/dy/cdyqfhhyat5mcsedgfuhvcwnymxdpawdvwpxvmppsgn7g5flvbl6.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsampled_ppm_out_1 => convert_element_type_5
# Graph fragment:
#   %convert_element_type_5 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzgloabegfohz7bkbxb3eeqiv3k5mpqyeeov2rzhwxfpdcwgagl.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out_1 => add_14, clamp_max_4
# Graph fragment:
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_5, 1), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_14, 1), kwargs = {})
triton_poi_fused_add_clamp_9 = async_compile.triton('triton_poi_fused_add_clamp_9', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = triton_helpers.minimum(tmp11, tmp10)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ic/cicsw5wxa2tq37wihwg46ha4uhigmbjom7rqdbzo2m7kzwc76zep.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out => add_6, convert_element_type, iota
#   upsampled_ppm_out_1 => clamp_max_6, clamp_min_4, clamp_min_6, mul_5, sub_7, sub_9
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_6 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.125), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_5, 0.5), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_7, 0.0), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_7), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_9, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_10 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_10', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4722jdhk6cx6tu7yjchjov7pbbx64mrrxq33tm2l3ke24os7c6.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsampled_ppm_out_1 => _unsafe_index_4, _unsafe_index_5, add_17, mul_7, sub_10
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_28, [None, None, %convert_element_type_5, %convert_element_type_7]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_28, [None, None, %convert_element_type_5, %clamp_max_5]), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %clamp_max_6), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_7), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_11 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 2*tmp4 + 4*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/2t/c2tzt44m777fhfn4p3ml2jfqcjhmxy6fetevxrxok6pg764mcaww.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsampled_ppm_out_2 => convert_element_type_9
# Graph fragment:
#   %convert_element_type_9 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_12 = async_compile.triton('triton_poi_fused__to_copy_12', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.1875
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/55/c55di7iz54zagjxtuc5jb5eymgwys2ljoexyrdmp5xkc6tcrwdqd.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out_2 => add_21, clamp_max_8
# Graph fragment:
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_9, 1), kwargs = {})
#   %clamp_max_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_21, 2), kwargs = {})
triton_poi_fused_add_clamp_13 = async_compile.triton('triton_poi_fused_add_clamp_13', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.1875
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7b/c7b35whq7o74wi2fmsha67vt3rwm4diwveb2dagr7ald5euy6oac.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_2], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out => add_6, convert_element_type, iota
#   upsampled_ppm_out_2 => clamp_max_10, clamp_min_10, clamp_min_8, mul_10, sub_14, sub_16
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_6 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.1875), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_10, 0.5), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_14, 0.0), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_11), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_16, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.1875
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/42/c427idawlfvg4uia5ailiaj4vb3cybazayktcotuv3pdjvo6n52b.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsampled_ppm_out_2 => _unsafe_index_8, _unsafe_index_9, add_24, mul_12, sub_17
# Graph fragment:
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_29, [None, None, %convert_element_type_9, %convert_element_type_11]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_29, [None, None, %convert_element_type_9, %clamp_max_9]), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %clamp_max_10), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_12), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_15 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 3, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 3*tmp4 + 9*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 3*tmp4 + 9*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/yc/cyc4dp4wam75jgqd5vwvczidayqayoywjpqh6m3u7nnpybc5j3rj.py
# Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   input_40 => _adaptive_avg_pool2d_2
# Graph fragment:
#   %_adaptive_avg_pool2d_2 : [num_users=2] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%add_5, [6, 6]), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_16 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (8*x1) // 3
    tmp1 = (21 + 16*x1) // 6
    tmp2 = tmp0 < tmp1
    tmp3 = (8*x0) // 3
    tmp4 = (21 + 16*x0) // 6
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = 1 + ((8*x0) // 3)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 2 + ((8*x0) // 3)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (2 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = 3 + ((8*x0) // 3)
    tmp19 = tmp18 < tmp4
    tmp20 = tmp2 & tmp19
    tmp21 = tl.load(in_ptr0 + (3 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = 1 + ((8*x1) // 3)
    tmp24 = tmp23 < tmp1
    tmp25 = tmp24 & tmp5
    tmp26 = tl.load(in_ptr0 + (16 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26 + tmp22
    tmp28 = tmp24 & tmp9
    tmp29 = tl.load(in_ptr0 + (17 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp27
    tmp31 = tmp24 & tmp14
    tmp32 = tl.load(in_ptr0 + (18 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp30
    tmp34 = tmp24 & tmp19
    tmp35 = tl.load(in_ptr0 + (19 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = 2 + ((8*x1) // 3)
    tmp38 = tmp37 < tmp1
    tmp39 = tmp38 & tmp5
    tmp40 = tl.load(in_ptr0 + (32 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 + tmp36
    tmp42 = tmp38 & tmp9
    tmp43 = tl.load(in_ptr0 + (33 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp43 + tmp41
    tmp45 = tmp38 & tmp14
    tmp46 = tl.load(in_ptr0 + (34 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp46 + tmp44
    tmp48 = tmp38 & tmp19
    tmp49 = tl.load(in_ptr0 + (35 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 + tmp47
    tmp51 = 3 + ((8*x1) // 3)
    tmp52 = tmp51 < tmp1
    tmp53 = tmp52 & tmp5
    tmp54 = tl.load(in_ptr0 + (48 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp53 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 + tmp50
    tmp56 = tmp52 & tmp9
    tmp57 = tl.load(in_ptr0 + (49 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp57 + tmp55
    tmp59 = tmp52 & tmp14
    tmp60 = tl.load(in_ptr0 + (50 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp60 + tmp58
    tmp62 = tmp52 & tmp19
    tmp63 = tl.load(in_ptr0 + (51 + 16*((8*x1) // 3) + 256*x2 + ((8*x0) // 3)), tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp64 = tmp63 + tmp61
    tmp65 = 1.0
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp6, tmp65, tmp66)
    tmp68 = 1.0
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp10, tmp68, tmp69)
    tmp71 = tmp70 + tmp67
    tmp72 = 1.0
    tmp73 = tl.full(tmp72.shape, 0.0, tmp72.dtype)
    tmp74 = tl.where(tmp15, tmp72, tmp73)
    tmp75 = tmp74 + tmp71
    tmp76 = 1.0
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp20, tmp76, tmp77)
    tmp79 = tmp78 + tmp75
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 0.0, tmp80.dtype)
    tmp82 = tl.where(tmp25, tmp80, tmp81)
    tmp83 = tmp82 + tmp79
    tmp84 = 1.0
    tmp85 = tl.full(tmp84.shape, 0.0, tmp84.dtype)
    tmp86 = tl.where(tmp28, tmp84, tmp85)
    tmp87 = tmp86 + tmp83
    tmp88 = 1.0
    tmp89 = tl.full(tmp88.shape, 0.0, tmp88.dtype)
    tmp90 = tl.where(tmp31, tmp88, tmp89)
    tmp91 = tmp90 + tmp87
    tmp92 = 1.0
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp34, tmp92, tmp93)
    tmp95 = tmp94 + tmp91
    tmp96 = 1.0
    tmp97 = tl.full(tmp96.shape, 0.0, tmp96.dtype)
    tmp98 = tl.where(tmp39, tmp96, tmp97)
    tmp99 = tmp98 + tmp95
    tmp100 = 1.0
    tmp101 = tl.full(tmp100.shape, 0.0, tmp100.dtype)
    tmp102 = tl.where(tmp42, tmp100, tmp101)
    tmp103 = tmp102 + tmp99
    tmp104 = 1.0
    tmp105 = tl.full(tmp104.shape, 0.0, tmp104.dtype)
    tmp106 = tl.where(tmp45, tmp104, tmp105)
    tmp107 = tmp106 + tmp103
    tmp108 = 1.0
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp48, tmp108, tmp109)
    tmp111 = tmp110 + tmp107
    tmp112 = 1.0
    tmp113 = tl.full(tmp112.shape, 0.0, tmp112.dtype)
    tmp114 = tl.where(tmp53, tmp112, tmp113)
    tmp115 = tmp114 + tmp111
    tmp116 = 1.0
    tmp117 = tl.full(tmp116.shape, 0.0, tmp116.dtype)
    tmp118 = tl.where(tmp56, tmp116, tmp117)
    tmp119 = tmp118 + tmp115
    tmp120 = 1.0
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp59, tmp120, tmp121)
    tmp123 = tmp122 + tmp119
    tmp124 = 1.0
    tmp125 = tl.full(tmp124.shape, 0.0, tmp124.dtype)
    tmp126 = tl.where(tmp62, tmp124, tmp125)
    tmp127 = tmp126 + tmp123
    tmp128 = tmp64 / tmp127
    tl.store(out_ptr0 + (x4), tmp128, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvrgkolgvyhyw56qjg4xpt7g7rmplmt3nz4rwt3rtwtqhxsibir.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   upsampled_ppm_out_3 => convert_element_type_13
# Graph fragment:
#   %convert_element_type_13 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
triton_poi_fused__to_copy_17 = async_compile.triton('triton_poi_fused__to_copy_17', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.375
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jk/cjkoaoso5jhyq63n3fepf2uidj5br2ribtlxstgfupywun2hais3.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out_3 => add_28, clamp_max_12
# Graph fragment:
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_13, 1), kwargs = {})
#   %clamp_max_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_28, 5), kwargs = {})
triton_poi_fused_add_clamp_18 = async_compile.triton('triton_poi_fused_add_clamp_18', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_18(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.375
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 5, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2r/c2rb2wbkqtar4iyh5peiq7y3gkrekba2uaxtmoxf55agjl6rk6pn.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   upsampled_ppm_out => add_6, convert_element_type, iota
#   upsampled_ppm_out_3 => clamp_max_14, clamp_min_12, clamp_min_14, mul_15, sub_21, sub_23
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_6 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.375), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_15, 0.5), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_21, 0.0), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_15), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_23, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.375
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aw/caw2zu62pqhgeankaavyayllnetga6gh5jimecbwvelm2jf4a7wx.py
# Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   upsampled_ppm_out_3 => _unsafe_index_12, _unsafe_index_13, add_31, mul_17, sub_24
# Graph fragment:
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_30, [None, None, %convert_element_type_13, %convert_element_type_15]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_30, [None, None, %convert_element_type_13, %clamp_max_13]), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %clamp_max_14), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_17), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_20 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 6, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 6*tmp4 + 36*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 6*tmp4 + 36*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x4), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/cv/ccvb5kj4j3a4oyi3sgozlfo7wgffbmvgntwtcrtat5k354q7urlq.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_5, %add_12, %add_19, %add_26, %add_33], 1), kwargs = {})
triton_poi_fused_cat_21 = async_compile.triton('triton_poi_fused_cat_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*i64', 'in_ptr18': '*fp32', 'in_ptr19': '*i64', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*i64', 'in_ptr24': '*i64', 'in_ptr25': '*fp32', 'in_ptr26': '*i64', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 256)
    x3 = xindex // 65536
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 32768*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 160, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 256*((-128) + x2) + 8192*x3), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([XBLOCK], 1, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tmp16 = tl.load(in_ptr3 + (x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr4 + (32*x3 + ((-128) + x2)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp12
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tmp20 - tmp20
    tmp26 = tl.load(in_ptr6 + (x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 + tmp27
    tmp29 = tmp28 - tmp10
    tmp30 = tl.load(in_ptr7 + (x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp10 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp9, tmp32, tmp33)
    tmp35 = tmp0 >= tmp7
    tmp36 = tl.full([1], 192, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tmp35 & tmp37
    tmp39 = tl.load(in_ptr8 + (x4 + 256*((-160) + x2) + 8192*x3), tmp38, other=0.0)
    tmp40 = tl.load(in_ptr9 + (x1), tmp38, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full([XBLOCK], 2, tl.int32)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp40 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp40)
    tmp45 = tl.load(in_ptr10 + (x0), tmp38, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tmp49 = tl.load(in_ptr11 + (tmp48 + 2*tmp44 + 4*((-160) + x2) + 128*x3), tmp38, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr12 + (x0), tmp38, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp41
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr11 + (tmp53 + 2*tmp44 + 4*((-160) + x2) + 128*x3), tmp38, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 - tmp49
    tmp56 = tl.load(in_ptr13 + (x0), tmp38, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tmp49 + tmp57
    tmp59 = tmp58 - tmp39
    tmp60 = tl.load(in_ptr14 + (x1), tmp38, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp39 + tmp61
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp38, tmp62, tmp63)
    tmp65 = tmp0 >= tmp36
    tmp66 = tl.full([1], 224, tl.int64)
    tmp67 = tmp0 < tmp66
    tmp68 = tmp65 & tmp67
    tmp69 = tl.load(in_ptr15 + (x4 + 256*((-192) + x2) + 8192*x3), tmp68, other=0.0)
    tmp70 = tl.load(in_ptr16 + (x1), tmp68, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.full([XBLOCK], 3, tl.int32)
    tmp72 = tmp70 + tmp71
    tmp73 = tmp70 < 0
    tmp74 = tl.where(tmp73, tmp72, tmp70)
    tmp75 = tl.load(in_ptr17 + (x0), tmp68, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp75 + tmp71
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tmp79 = tl.load(in_ptr18 + (tmp78 + 3*tmp74 + 9*((-192) + x2) + 288*x3), tmp68, eviction_policy='evict_last', other=0.0)
    tmp80 = tl.load(in_ptr19 + (x0), tmp68, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp71
    tmp82 = tmp80 < 0
    tmp83 = tl.where(tmp82, tmp81, tmp80)
    tmp84 = tl.load(in_ptr18 + (tmp83 + 3*tmp74 + 9*((-192) + x2) + 288*x3), tmp68, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp84 - tmp79
    tmp86 = tl.load(in_ptr20 + (x0), tmp68, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 * tmp86
    tmp88 = tmp79 + tmp87
    tmp89 = tmp88 - tmp69
    tmp90 = tl.load(in_ptr21 + (x1), tmp68, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 * tmp90
    tmp92 = tmp69 + tmp91
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp68, tmp92, tmp93)
    tmp95 = tmp0 >= tmp66
    tmp96 = tl.full([1], 256, tl.int64)
    tmp97 = tmp0 < tmp96
    tmp98 = tl.load(in_ptr22 + (x4 + 256*((-224) + x2) + 8192*x3), tmp95, other=0.0)
    tmp99 = tl.load(in_ptr23 + (x1), tmp95, eviction_policy='evict_last', other=0.0)
    tmp100 = tl.full([XBLOCK], 6, tl.int32)
    tmp101 = tmp99 + tmp100
    tmp102 = tmp99 < 0
    tmp103 = tl.where(tmp102, tmp101, tmp99)
    tmp104 = tl.load(in_ptr24 + (x0), tmp95, eviction_policy='evict_last', other=0.0)
    tmp105 = tmp104 + tmp100
    tmp106 = tmp104 < 0
    tmp107 = tl.where(tmp106, tmp105, tmp104)
    tmp108 = tl.load(in_ptr25 + (tmp107 + 6*tmp103 + 36*((-224) + x2) + 1152*x3), tmp95, eviction_policy='evict_last', other=0.0)
    tmp109 = tl.load(in_ptr26 + (x0), tmp95, eviction_policy='evict_last', other=0.0)
    tmp110 = tmp109 + tmp100
    tmp111 = tmp109 < 0
    tmp112 = tl.where(tmp111, tmp110, tmp109)
    tmp113 = tl.load(in_ptr25 + (tmp112 + 6*tmp103 + 36*((-224) + x2) + 1152*x3), tmp95, eviction_policy='evict_last', other=0.0)
    tmp114 = tmp113 - tmp108
    tmp115 = tl.load(in_ptr27 + (x0), tmp95, eviction_policy='evict_last', other=0.0)
    tmp116 = tmp114 * tmp115
    tmp117 = tmp108 + tmp116
    tmp118 = tmp117 - tmp98
    tmp119 = tl.load(in_ptr28 + (x1), tmp95, eviction_policy='evict_last', other=0.0)
    tmp120 = tmp118 * tmp119
    tmp121 = tmp98 + tmp120
    tmp122 = tl.full(tmp121.shape, 0.0, tmp121.dtype)
    tmp123 = tl.where(tmp95, tmp121, tmp122)
    tmp124 = tl.where(tmp68, tmp94, tmp123)
    tmp125 = tl.where(tmp38, tmp64, tmp124)
    tmp126 = tl.where(tmp9, tmp34, tmp125)
    tmp127 = tl.where(tmp4, tmp5, tmp126)
    tl.store(out_ptr0 + (x5), tmp127, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33 = args
    args.clear()
    assert_size_stride(primals_1, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_2, (4, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_3, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_4, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_5, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_6, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_7, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_8, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_9, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_10, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_11, (384, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_12, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_13, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_14, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_15, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_16, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_17, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_18, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_19, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_20, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_21, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_22, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_23, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_24, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_25, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_26, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_27, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_28, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_29, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_30, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_31, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_32, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_33, (128, 256, 3, 3), (2304, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 384, 64, 64), (1572864, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf1, (4, 384, 32, 32), (393216, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 32, 32), (65536, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 384, 32, 32), (393216, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf4, (4, 384, 32, 32), (393216, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(buf6, buf2, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 384, 32, 32), (393216, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf8, (4, 384, 32, 32), (393216, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(buf10, buf6, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 384, 32, 32), (393216, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf12, (4, 384, 16, 16), (98304, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 96, 16, 16), (24576, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 576, 16, 16), (147456, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf15, (4, 576, 16, 16), (147456, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf17, buf13, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 576, 16, 16), (147456, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf19, (4, 576, 16, 16), (147456, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf21, buf17, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 576, 16, 16), (147456, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf23, (4, 576, 16, 16), (147456, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 128, 16, 16), (32768, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 768, 16, 16), (196608, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf26, (4, 768, 16, 16), (196608, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf28, buf24, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 768, 16, 16), (196608, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf30, (4, 768, 16, 16), (196608, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf32 = buf31; del buf31  # reuse
        buf33 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf34 = reinterpret_tensor(buf33, (4, 128, 1, 1), (128, 1, 1, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_3.run(buf32, buf34, buf28, 512, 256, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 32, 1, 1), (32, 1, 1, 1))
        buf36 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf36, 16, grid=grid(16), stream=stream0)
        buf37 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_5.run(buf37, 16, grid=grid(16), stream=stream0)
        buf38 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf38, 16, grid=grid(16), stream=stream0)
        buf39 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_5.run(buf39, 16, grid=grid(16), stream=stream0)
        buf40 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_6.run(buf40, 16, grid=grid(16), stream=stream0)
        buf41 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_7.run(buf36, buf38, buf35, buf39, buf40, buf41, 32768, grid=grid(32768), stream=stream0)
        buf42 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_6.run(buf42, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten._adaptive_avg_pool2d]
        buf43 = torch.ops.aten.avg_pool2d.default(buf32, [8, 8], [8, 8], [0, 0], False, True, None)
        buf44 = buf43
        del buf43
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 32, 2, 2), (128, 4, 2, 1))
        buf46 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(buf46, 16, grid=grid(16), stream=stream0)
        buf47 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_9.run(buf47, 16, grid=grid(16), stream=stream0)
        buf48 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(buf48, 16, grid=grid(16), stream=stream0)
        buf49 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_9.run(buf49, 16, grid=grid(16), stream=stream0)
        buf50 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_10.run(buf50, 16, grid=grid(16), stream=stream0)
        buf51 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_11.run(buf46, buf48, buf45, buf49, buf50, buf51, 32768, grid=grid(32768), stream=stream0)
        buf52 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_10.run(buf52, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten._adaptive_avg_pool2d]
        buf53 = torch.ops.aten._adaptive_avg_pool2d.default(buf32, [3, 3])
        buf54 = buf53
        del buf53
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 32, 3, 3), (288, 9, 3, 1))
        buf56 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf56, 16, grid=grid(16), stream=stream0)
        buf57 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_13.run(buf57, 16, grid=grid(16), stream=stream0)
        buf58 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_2], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_12.run(buf58, 16, grid=grid(16), stream=stream0)
        buf59 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_13.run(buf59, 16, grid=grid(16), stream=stream0)
        buf60 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_2], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14.run(buf60, 16, grid=grid(16), stream=stream0)
        buf61 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_15.run(buf56, buf58, buf55, buf59, buf60, buf61, 32768, grid=grid(32768), stream=stream0)
        buf62 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_2], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_14.run(buf62, 16, grid=grid(16), stream=stream0)
        buf63 = empty_strided_cuda((4, 128, 6, 6), (4608, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._adaptive_avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_16.run(buf32, buf63, 18432, grid=grid(18432), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 32, 6, 6), (1152, 36, 6, 1))
        buf65 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf65, 16, grid=grid(16), stream=stream0)
        buf66 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_18.run(buf66, 16, grid=grid(16), stream=stream0)
        buf67 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_17.run(buf67, 16, grid=grid(16), stream=stream0)
        buf68 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_18.run(buf68, 16, grid=grid(16), stream=stream0)
        buf69 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out, upsampled_ppm_out_3], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19.run(buf69, 16, grid=grid(16), stream=stream0)
        buf70 = empty_strided_cuda((4, 32, 16, 16), (8192, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_20.run(buf65, buf67, buf64, buf68, buf69, buf70, 32768, grid=grid(32768), stream=stream0)
        buf71 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [upsampled_ppm_out_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_19.run(buf71, 16, grid=grid(16), stream=stream0)
        buf72 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_21.run(buf32, buf41, buf37, buf38, buf35, buf39, buf40, buf42, buf51, buf47, buf48, buf45, buf49, buf50, buf52, buf61, buf57, buf58, buf55, buf59, buf60, buf62, buf70, buf66, buf67, buf64, buf68, buf69, buf71, buf72, 262144, grid=grid(262144), stream=stream0)
        del buf35
        del buf41
        del buf45
        del buf51
        del buf55
        del buf61
        del buf64
        del buf70
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 128, 16, 16), (32768, 256, 16, 1))
    return (buf73, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, buf0, buf1, buf2, buf3, buf4, buf6, buf7, buf8, buf10, buf11, buf12, buf13, buf14, buf15, buf17, buf18, buf19, buf21, buf22, buf23, buf24, buf25, buf26, buf28, buf29, buf30, buf32, buf34, buf36, buf37, buf38, buf39, buf40, buf42, buf44, buf46, buf47, buf48, buf49, buf50, buf52, buf54, buf56, buf57, buf58, buf59, buf60, buf62, buf63, buf65, buf66, buf67, buf68, buf69, buf71, buf72, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
