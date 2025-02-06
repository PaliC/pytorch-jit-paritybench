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


# kernel path: inductor_cache/5b/c5bc35vvmpdiog76nlgag7q2k2jp7pz5ds74af276kiset36ecin.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 3)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ne/cnebdx3c6msta2w6ju7eomdtzfdgi2in24v3g2pyfld7l7kjrwh2.py
# Topologically Sorted Source Nodes: [x1tcw], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   x1tcw => relu
# Graph fragment:
#   %relu : [num_users=5] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
triton_poi_fused_relu_1 = async_compile.triton('triton_poi_fused_relu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_1(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/jz/cjzbusrssjuxfra62ndlgqb3eclzzxieyrpfw2oe365bpz3fecn6.py
# Topologically Sorted Source Nodes: [c1, c1_1], Original ATen: [aten.cat, aten.relu]
# Source node to ATen node mapping:
#   c1 => cat
#   c1_1 => relu_2
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %convolution_3], 1), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%cat,), kwargs = {})
triton_poi_fused_cat_relu_2 = async_compile.triton('triton_poi_fused_cat_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_relu_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 128)
    x0 = (xindex % 4096)
    x2 = xindex // 524288
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4096*(x1) + 262144*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 4096*((-64) + x1) + 262144*x2), tmp6, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tl.store(out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/u5/cu5ircb6ayensnljaf5jfrcah4ydlth5ia5hq6htsux5gflj55yf.py
# Topologically Sorted Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15, x34, x34_1], Original ATen: [aten.add, aten.relu]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_10 => add_10
#   add_11 => add_11
#   add_12 => add_12
#   add_13 => add_13
#   add_14 => add_14
#   add_15 => add_15
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   add_7 => add_7
#   add_8 => add_8
#   add_9 => add_9
#   x34 => add_16
#   x34_1 => relu_33
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %convolution_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %convolution_5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %convolution_7), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %convolution_9), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %convolution_11), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_13), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %convolution_15), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %convolution_17), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_7, %convolution_19), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %convolution_21), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_23), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %convolution_25), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %convolution_27), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_12, %convolution_29), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %convolution_31), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %convolution_33), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %convolution_34), kwargs = {})
#   %relu_33 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
triton_poi_fused_add_relu_3 = async_compile.triton('triton_poi_fused_add_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 18, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_out_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x0), None)
    tmp11 = tl.load(in_ptr5 + (x0), None)
    tmp13 = tl.load(in_ptr6 + (x0), None)
    tmp15 = tl.load(in_ptr7 + (x0), None)
    tmp17 = tl.load(in_ptr8 + (x0), None)
    tmp19 = tl.load(in_ptr9 + (x0), None)
    tmp21 = tl.load(in_ptr10 + (x0), None)
    tmp23 = tl.load(in_ptr11 + (x0), None)
    tmp25 = tl.load(in_ptr12 + (x0), None)
    tmp27 = tl.load(in_ptr13 + (x0), None)
    tmp29 = tl.load(in_ptr14 + (x0), None)
    tmp31 = tl.load(in_ptr15 + (x0), None)
    tmp33 = tl.load(in_ptr16 + (x0), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tl.store(in_out_ptr0 + (x0), tmp36, None)
''', device_str='cuda')


# kernel path: inductor_cache/lv/clvyyejhpsjwpm2c3qsyhnoubwwn7txp7ne226eh2oahlkeig3hb.py
# Topologically Sorted Source Nodes: [input_60, temp1, temp2], Original ATen: [aten.relu, aten.add, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_60 => relu_38
#   temp1 => add_17
#   temp2 => relu_39
# Graph fragment:
#   %relu_38 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_39,), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %relu_38), kwargs = {})
#   %relu_39 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   %le_6 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_38, 0), kwargs = {})
triton_poi_fused_add_relu_threshold_backward_4 = async_compile.triton('triton_poi_fused_add_relu_threshold_backward_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_relu_threshold_backward_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_relu_threshold_backward_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = triton_helpers.maximum(tmp2, tmp1)
    tmp4 = tmp0 + tmp3
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp6 = 0.0
    tmp7 = tmp3 <= tmp6
    tl.store(out_ptr0 + (x0), tmp5, None)
    tl.store(out_ptr1 + (x0), tmp7, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50 = args
    args.clear()
    assert_size_stride(primals_1, (3, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (3, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_5, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_6, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_8, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_10, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_13, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_14, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_18, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_19, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_20, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_21, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_24, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_26, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_27, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_30, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_31, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_32, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_33, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_34, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_36, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_37, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_38, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_39, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_40, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_41, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_42, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_43, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_44, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_45, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_46, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_47, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_48, (3, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_49, (3, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_50, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 3, 64, 64), (12288, 4096, 64, 1))
        del primals_1
        del primals_3
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf1, primals_2, 49152, grid=grid(49152), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x1tcw], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf3, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf5, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf7 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c1, c1_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf3, buf6, buf7, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf9, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf11 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c2, c2_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf6, buf10, buf11, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf13, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf15 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c3, c3_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf10, buf14, buf15, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf17, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf19 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c4, c4_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf14, buf18, buf19, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf21, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf23 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c5, c5_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf18, buf22, buf23, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf25, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf27 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c6, c6_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf22, buf26, buf27, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf29, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf31 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c7, c7_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf26, buf30, buf31, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf33, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf35 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c7_2, c7_3], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf30, buf34, buf35, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf37, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf39 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c8, c8_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf34, buf38, buf39, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf41 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf41, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf43 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c9, c9_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf38, buf42, buf43, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf45, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf47 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c10, c10_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf42, buf46, buf47, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf49 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf49, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf51 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c11, c11_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf46, buf50, buf51, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf53, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf55 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c12, c12_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf50, buf54, buf55, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf57, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf59 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c13, c13_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf54, buf58, buf59, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf61, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf63 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c14, c14_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf58, buf62, buf63, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf65, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf67 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [c15, c15_1], Original ATen: [aten.cat, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_relu_2.run(buf62, buf66, buf67, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf69 = buf6; del buf6  # reuse
        buf70 = buf69; del buf69  # reuse
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [add, add_1, add_2, add_3, add_4, add_5, add_6, add_7, add_8, add_9, add_10, add_11, add_12, add_13, add_14, add_15, x34, x34_1], Original ATen: [aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_relu_3.run(buf71, buf3, buf10, buf14, buf18, buf22, buf26, buf30, buf34, buf38, buf42, buf46, buf50, buf54, buf58, buf62, buf66, buf68, 1048576, grid=grid(1048576), stream=stream0)
        del buf10
        del buf14
        del buf18
        del buf22
        del buf26
        del buf30
        del buf34
        del buf38
        del buf42
        del buf46
        del buf50
        del buf54
        del buf58
        del buf62
        del buf66
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf73, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf75, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf77, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf79, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf81 = buf68; del buf68  # reuse
        buf95 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_60, temp1, temp2], Original ATen: [aten.relu, aten.add, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_relu_threshold_backward_4.run(buf3, buf80, buf81, buf95, 1048576, grid=grid(1048576), stream=stream0)
        del buf80
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [temp3_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf83, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [temp4_2], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf85, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_45, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [temp5_2], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf87, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [temp6_2], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf89, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [temp7_1], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_1.run(buf91, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 3, 64, 64), (12288, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 3, 64, 64), (12288, 4096, 64, 1))
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf94, primals_50, 49152, grid=grid(49152), stream=stream0)
        del primals_50
    return (buf94, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, buf1, buf3, buf5, buf7, buf9, buf11, buf13, buf15, buf17, buf19, buf21, buf23, buf25, buf27, buf29, buf31, buf33, buf35, buf37, buf39, buf41, buf43, buf45, buf47, buf49, buf51, buf53, buf55, buf57, buf59, buf61, buf63, buf65, buf67, buf71, buf73, buf75, buf77, buf79, buf81, buf83, buf85, buf87, buf89, buf91, buf92, buf95, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((3, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((3, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
