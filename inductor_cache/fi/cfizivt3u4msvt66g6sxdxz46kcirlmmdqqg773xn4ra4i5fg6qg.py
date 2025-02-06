# AOT ID: ['8_forward']
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


# kernel path: inductor_cache/mi/cmilbcz2m6d6im7pfgr22264hps745rahzp67kncfwto7ox6ngqm.py
# Topologically Sorted Source Nodes: [out1, out1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out1 => add_1, mul_1, mul_2, sub
#   out1_1 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/le/clea573nnd5gmgkamikcety7klydk7rnkxd4bedy5gexiwpptnfr.py
# Topologically Sorted Source Nodes: [out2, out2_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out2 => add_3, mul_4, mul_5, sub_1
#   out2_1 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v4/cv4vid4dhcl5h7mjqb2gpfccvva5l3hrnlumbxsbqcenlcvyiqoj.py
# Topologically Sorted Source Nodes: [out3, out3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out3 => add_5, mul_7, mul_8, sub_2
#   out3_1 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hz/chzl4k2f7hfub365w3w73lsfkui6zrswxmzbwqw32ahp7hcufa5x.py
# Topologically Sorted Source Nodes: [low1, out1_3, out1_4], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   low1 => avg_pool2d
#   out1_3 => add_8, mul_10, mul_11, sub_3
#   out1_4 => relu_3
# Graph fragment:
#   %avg_pool2d : [num_users=3] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%primals_1, [2, 2], [2, 2]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x5 = xindex
    x3 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_ptr0 + (2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
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
    tl.store(out_ptr0 + (x5), tmp8, xmask)
    tl.store(out_ptr1 + (x5), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ft/cftdp3fejxpw2qsubhnp6kada3hq73ykhnjvnt27uu7mub23jtu7.py
# Topologically Sorted Source Nodes: [out2_3, out2_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out2_3 => add_10, mul_13, mul_14, sub_4
#   out2_4 => relu_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 2)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pw/cpw7r6swrijeijstfo5w5hatorbx5gs6kfxwmxtgd36kjee253k2.py
# Topologically Sorted Source Nodes: [out3_5, out3_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out3_5 => add_12, mul_16, mul_17, sub_5
#   out3_6 => relu_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp3 * tmp12
    tmp16 = tmp13 * tmp15
    tmp19 = tmp16 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/av/cavlfwdxakmadrwysnetyszlscd7d4rrkbfnnbtu5i2pmcq2esou.py
# Topologically Sorted Source Nodes: [out3_8, out3_9, out1_6, out1_7], Original ATen: [aten.cat, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   out1_6 => add_15, mul_19, mul_20, sub_6
#   out1_7 => relu_6
#   out3_8 => cat_1
#   out3_9 => add_13
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_3, %convolution_4, %convolution_5], 1), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %avg_pool2d), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x3), xmask)
    tmp19 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 8*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 3, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 4*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 - tmp19
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tl.full([1], 1, tl.int32)
    tmp26 = tmp25 / tmp24
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tl.full([1], 0, tl.int32)
    tmp35 = triton_helpers.maximum(tmp34, tmp33)
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/py/cpy47qcn3blsf42sfrp3ko7nhpnquoyzg2iv234f2zhyeoui7y3f.py
# Topologically Sorted Source Nodes: [out3_9, out3_13, out3_14, out1_9, out1_10], Original ATen: [aten.add, aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   out1_10 => relu_9
#   out1_9 => add_22, mul_28, mul_29, sub_9
#   out3_13 => cat_2
#   out3_14 => add_20
#   out3_9 => add_13
# Graph fragment:
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_1, %avg_pool2d), kwargs = {})
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_6, %convolution_7, %convolution_8], 1), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_2, %add_13), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_20, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_22,), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_20, %unsqueeze_123), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_cat_native_batch_norm_backward_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_cat_native_batch_norm_backward_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_cat_native_batch_norm_backward_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_cat_native_batch_norm_backward_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp17 = tl.load(in_ptr3 + (x3), xmask)
    tmp18 = tl.load(in_ptr4 + (x3), xmask)
    tmp21 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 4*(x1) + 8*x2), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 3, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 4*x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + (x0 + 4*x2), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp22 = tmp20 - tmp21
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.sqrt(tmp25)
    tmp27 = tl.full([1], 1, tl.int32)
    tmp28 = tmp27 / tmp26
    tmp29 = 1.0
    tmp30 = tmp28 * tmp29
    tmp31 = tmp22 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tl.full([1], 0, tl.int32)
    tmp37 = triton_helpers.maximum(tmp36, tmp35)
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tl.store(out_ptr1 + (x3), tmp37, xmask)
    tl.store(out_ptr2 + (x3), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5a/c5aww2zmku5sk2sdssk2yqxpzuahrlnbmvsv2gky6tj4jp7u37c2.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   up2 => clamp_max_2, clamp_min_2, convert_element_type_27, floor_1, sub_14
# Graph fragment:
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_96,), kwargs = {})
#   %convert_element_type_27 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_27, 1), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_14, 0), kwargs = {})
#   %clamp_max_2 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1), kwargs = {})
triton_poi_fused__to_copy_clamp_floor_sub_8 = async_compile.triton('triton_poi_fused__to_copy_clamp_floor_sub_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_floor_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_floor_sub_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = triton_helpers.minimum(tmp9, tmp6)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lk/clkrej47cqckq5552hvlfpbz7t3bciuyo3zwpl45kkcicp6ino34.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   up2 => clamp_max_5, clamp_min_5, convert_element_type_24, convert_element_type_26, floor, iota, mul_36
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_36 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, 0.3333333333333333), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_36,), kwargs = {})
#   %convert_element_type_26 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_26, 0), kwargs = {})
#   %clamp_max_5 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 1), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_floor_mul_9 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_floor_mul_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_floor_mul_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_floor_mul_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l6/cl6fwtgnbbx7tid3wdedzdzj6dsfdgixxkgqlp7iylnmd27ku7si.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up2 => add_30, clamp_max_7, clamp_min_7, convert_element_type_24, convert_element_type_26, floor, iota, mul_36
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_36 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, 0.3333333333333333), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_36,), kwargs = {})
#   %convert_element_type_26 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_26, 1), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_30, 0), kwargs = {})
#   %clamp_max_7 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 1), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_10 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = triton_helpers.minimum(tmp9, tmp6)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t3/ct3fijyaiiy76nqlzcqjxgaf574h5tklotrgesg2r3obm44tjb6n.py
# Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   up2 => add_31, clamp_max_9, clamp_min_9, convert_element_type_24, convert_element_type_26, floor, iota, mul_36
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_36 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, 0.3333333333333333), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_36,), kwargs = {})
#   %convert_element_type_26 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_26, 2), kwargs = {})
#   %clamp_min_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_31, 0), kwargs = {})
#   %clamp_max_9 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_9, 1), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_11 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_11(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.floor(tmp3)
    tmp5 = tmp4.to(tl.int32)
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rd/crd43w3dtcmtxwmd6iwnf2p455atepbdbjza7eumbmv5xsw4epul.py
# Topologically Sorted Source Nodes: [out3_3, out3_4, out3_18, out3_19, up2, add], Original ATen: [aten.cat, aten.add, aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.rsub, aten._unsafe_index]
# Source node to ATen node mapping:
#   add => add_57
#   out3_18 => cat_3
#   out3_19 => add_27
#   out3_3 => cat
#   out3_4 => add_6
#   up2 => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add_32, add_33, add_34, add_35, add_36, add_37, add_38, add_39, add_40, add_41, add_42, add_43, add_44, add_45, add_46, add_47, add_48, add_49, add_50, add_51, add_52, add_53, add_54, add_55, add_56, clamp_max, clamp_max_1, clamp_min, clamp_min_1, convert_element_type_24, floor, floor_1, iota, mul_36, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_46, mul_47, mul_48, mul_49, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, mul_58, mul_59, mul_60, mul_61, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, mul_81, sub_12, sub_13, sub_16, sub_17, sub_18, sub_19, sub_20, sub_21, sub_22, sub_23, sub_24, sub_25, sub_26, sub_27, sub_28, sub_29, sub_30, sub_31
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution, %convolution_1, %convolution_2], 1), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %primals_1), kwargs = {})
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_9, %convolution_10, %convolution_11], 1), kwargs = {})
#   %add_27 : [num_users=16] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_3, %add_20), kwargs = {})
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_36 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, 0.3333333333333333), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%mul_36,), kwargs = {})
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze_96,), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_96, %floor_1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_12, 0.0), kwargs = {})
#   %clamp_max : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_36, %floor), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_13, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1.0), kwargs = {})
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_1, 1.0), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, -0.75), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_38, -3.75), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %add_32), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_39, -6.0), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_33, %add_32), kwargs = {})
#   %sub_17 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_40, -3.0), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 1.25), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_41, 2.25), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %clamp_max_1), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %clamp_max_1), kwargs = {})
#   %add_34 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, 1), kwargs = {})
#   %sub_19 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_1), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, 1.25), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_44, 2.25), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %sub_19), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_45, %sub_19), kwargs = {})
#   %add_35 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, 1), kwargs = {})
#   %sub_21 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_1), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, -0.75), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_47, -3.75), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %sub_21), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, -6.0), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %sub_21), kwargs = {})
#   %sub_23 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_49, -3.0), kwargs = {})
#   %add_37 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, 1.0), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, -0.75), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_50, -3.75), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %add_37), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_51, -6.0), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, %add_37), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_52, -3.0), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 1.25), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_53, 2.25), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %clamp_max), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %clamp_max), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, 1), kwargs = {})
#   %sub_27 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, 1.25), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_56, 2.25), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %sub_27), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %sub_27), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_58, 1), kwargs = {})
#   %sub_29 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, -0.75), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_59, -3.75), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %sub_29), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_60, -6.0), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %sub_29), kwargs = {})
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_61, -3.0), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_2, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_2, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_2, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_2, %clamp_max_9]), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, %sub_17), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_1, %add_34), kwargs = {})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %mul_63), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_2, %add_35), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_42, %mul_64), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_3, %sub_23), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_43, %mul_65), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_10, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_10, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_10, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_10, %clamp_max_9]), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_4, %sub_17), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_5, %add_34), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %mul_67), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_6, %add_35), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %mul_68), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_7, %sub_23), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_46, %mul_69), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_18, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_18, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_18, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_18, %clamp_max_9]), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_8, %sub_17), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_9, %add_34), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %mul_71), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_10, %add_35), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_48, %mul_72), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_11, %sub_23), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %mul_73), kwargs = {})
#   %_unsafe_index_12 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_26, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_26, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_14 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_26, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_27, [None, None, %clamp_max_26, %clamp_max_9]), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_12, %sub_17), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_13, %add_34), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %mul_75), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_14, %add_35), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_51, %mul_76), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_15, %sub_23), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_52, %mul_77), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_44, %sub_25), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %add_39), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_78, %mul_79), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_50, %add_40), kwargs = {})
#   %add_55 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %mul_80), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sub_31), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %mul_81), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %add_56), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_rsub_sub_12 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_rsub_sub_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_rsub_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_rsub_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex // 16
    x7 = xindex
    x5 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp179 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp271 = tl.load(in_ptr15 + (x7), xmask)
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = x2
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp12
    tmp16 = tl.full([1], 3, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr3 + (tmp8 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp9 >= tmp16
    tmp21 = tl.full([1], 4, tl.int64)
    tmp22 = tmp9 < tmp21
    tmp23 = tl.load(in_ptr4 + (tmp8 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp19, tmp23)
    tmp25 = tl.where(tmp13, tmp14, tmp24)
    tmp26 = tl.load(in_ptr5 + (tmp8 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp27 = tmp25 + tmp26
    tmp28 = x0
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 0.3333333333333333
    tmp31 = tmp29 * tmp30
    tmp32 = libdevice.floor(tmp31)
    tmp33 = tmp31 - tmp32
    tmp34 = 0.0
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = 1.0
    tmp37 = triton_helpers.minimum(tmp35, tmp36)
    tmp38 = tmp37 + tmp36
    tmp39 = -0.75
    tmp40 = tmp38 * tmp39
    tmp41 = -3.75
    tmp42 = tmp40 - tmp41
    tmp43 = tmp42 * tmp38
    tmp44 = -6.0
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45 * tmp38
    tmp47 = -3.0
    tmp48 = tmp46 - tmp47
    tmp49 = tmp27 * tmp48
    tmp51 = tmp50 + tmp1
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr2 + (tmp53 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tl.load(in_ptr3 + (tmp53 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr4 + (tmp53 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.where(tmp18, tmp55, tmp56)
    tmp58 = tl.where(tmp13, tmp54, tmp57)
    tmp59 = tl.load(in_ptr5 + (tmp53 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp60 = tmp58 + tmp59
    tmp61 = 1.25
    tmp62 = tmp37 * tmp61
    tmp63 = 2.25
    tmp64 = tmp62 - tmp63
    tmp65 = tmp64 * tmp37
    tmp66 = tmp65 * tmp37
    tmp67 = tmp66 + tmp36
    tmp68 = tmp60 * tmp67
    tmp70 = tmp69 + tmp1
    tmp71 = tmp69 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp69)
    tmp73 = tl.load(in_ptr2 + (tmp72 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.load(in_ptr3 + (tmp72 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tl.load(in_ptr4 + (tmp72 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.where(tmp18, tmp74, tmp75)
    tmp77 = tl.where(tmp13, tmp73, tmp76)
    tmp78 = tl.load(in_ptr5 + (tmp72 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp79 = tmp77 + tmp78
    tmp80 = tmp36 - tmp37
    tmp81 = tmp80 * tmp61
    tmp82 = tmp81 - tmp63
    tmp83 = tmp82 * tmp80
    tmp84 = tmp83 * tmp80
    tmp85 = tmp84 + tmp36
    tmp86 = tmp79 * tmp85
    tmp88 = tmp87 + tmp1
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tmp91 = tl.load(in_ptr2 + (tmp90 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp92 = tl.load(in_ptr3 + (tmp90 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tl.load(in_ptr4 + (tmp90 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp94 = tl.where(tmp18, tmp92, tmp93)
    tmp95 = tl.where(tmp13, tmp91, tmp94)
    tmp96 = tl.load(in_ptr5 + (tmp90 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp97 = tmp95 + tmp96
    tmp98 = 2.0
    tmp99 = tmp98 - tmp37
    tmp100 = tmp99 * tmp39
    tmp101 = tmp100 - tmp41
    tmp102 = tmp101 * tmp99
    tmp103 = tmp102 + tmp44
    tmp104 = tmp103 * tmp99
    tmp105 = tmp104 - tmp47
    tmp106 = tmp97 * tmp105
    tmp108 = tmp107 + tmp1
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tl.load(in_ptr2 + (tmp8 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp112 = tl.load(in_ptr3 + (tmp8 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tl.load(in_ptr4 + (tmp8 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tl.where(tmp18, tmp112, tmp113)
    tmp115 = tl.where(tmp13, tmp111, tmp114)
    tmp116 = tl.load(in_ptr5 + (tmp8 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp117 = tmp115 + tmp116
    tmp118 = tmp117 * tmp48
    tmp119 = tl.load(in_ptr2 + (tmp53 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp120 = tl.load(in_ptr3 + (tmp53 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp121 = tl.load(in_ptr4 + (tmp53 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp122 = tl.where(tmp18, tmp120, tmp121)
    tmp123 = tl.where(tmp13, tmp119, tmp122)
    tmp124 = tl.load(in_ptr5 + (tmp53 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp125 = tmp123 + tmp124
    tmp126 = tmp125 * tmp67
    tmp127 = tl.load(in_ptr2 + (tmp72 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp128 = tl.load(in_ptr3 + (tmp72 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp129 = tl.load(in_ptr4 + (tmp72 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp130 = tl.where(tmp18, tmp128, tmp129)
    tmp131 = tl.where(tmp13, tmp127, tmp130)
    tmp132 = tl.load(in_ptr5 + (tmp72 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp133 = tmp131 + tmp132
    tmp134 = tmp133 * tmp85
    tmp135 = tl.load(in_ptr2 + (tmp90 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr3 + (tmp90 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = tl.load(in_ptr4 + (tmp90 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp138 = tl.where(tmp18, tmp136, tmp137)
    tmp139 = tl.where(tmp13, tmp135, tmp138)
    tmp140 = tl.load(in_ptr5 + (tmp90 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp141 = tmp139 + tmp140
    tmp142 = tmp141 * tmp105
    tmp144 = tmp143 + tmp1
    tmp145 = tmp143 < 0
    tmp146 = tl.where(tmp145, tmp144, tmp143)
    tmp147 = tl.load(in_ptr2 + (tmp8 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp148 = tl.load(in_ptr3 + (tmp8 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp149 = tl.load(in_ptr4 + (tmp8 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp150 = tl.where(tmp18, tmp148, tmp149)
    tmp151 = tl.where(tmp13, tmp147, tmp150)
    tmp152 = tl.load(in_ptr5 + (tmp8 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp153 = tmp151 + tmp152
    tmp154 = tmp153 * tmp48
    tmp155 = tl.load(in_ptr2 + (tmp53 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp156 = tl.load(in_ptr3 + (tmp53 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp157 = tl.load(in_ptr4 + (tmp53 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp158 = tl.where(tmp18, tmp156, tmp157)
    tmp159 = tl.where(tmp13, tmp155, tmp158)
    tmp160 = tl.load(in_ptr5 + (tmp53 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp161 = tmp159 + tmp160
    tmp162 = tmp161 * tmp67
    tmp163 = tl.load(in_ptr2 + (tmp72 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp164 = tl.load(in_ptr3 + (tmp72 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp165 = tl.load(in_ptr4 + (tmp72 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp166 = tl.where(tmp18, tmp164, tmp165)
    tmp167 = tl.where(tmp13, tmp163, tmp166)
    tmp168 = tl.load(in_ptr5 + (tmp72 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp169 = tmp167 + tmp168
    tmp170 = tmp169 * tmp85
    tmp171 = tl.load(in_ptr2 + (tmp90 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp172 = tl.load(in_ptr3 + (tmp90 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp173 = tl.load(in_ptr4 + (tmp90 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp174 = tl.where(tmp18, tmp172, tmp173)
    tmp175 = tl.where(tmp13, tmp171, tmp174)
    tmp176 = tl.load(in_ptr5 + (tmp90 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp177 = tmp175 + tmp176
    tmp178 = tmp177 * tmp105
    tmp180 = tmp179 + tmp1
    tmp181 = tmp179 < 0
    tmp182 = tl.where(tmp181, tmp180, tmp179)
    tmp183 = tl.load(in_ptr2 + (tmp8 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp184 = tl.load(in_ptr3 + (tmp8 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp185 = tl.load(in_ptr4 + (tmp8 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp186 = tl.where(tmp18, tmp184, tmp185)
    tmp187 = tl.where(tmp13, tmp183, tmp186)
    tmp188 = tl.load(in_ptr5 + (tmp8 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp189 = tmp187 + tmp188
    tmp190 = tmp189 * tmp48
    tmp191 = tl.load(in_ptr2 + (tmp53 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp192 = tl.load(in_ptr3 + (tmp53 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp193 = tl.load(in_ptr4 + (tmp53 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp194 = tl.where(tmp18, tmp192, tmp193)
    tmp195 = tl.where(tmp13, tmp191, tmp194)
    tmp196 = tl.load(in_ptr5 + (tmp53 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp197 = tmp195 + tmp196
    tmp198 = tmp197 * tmp67
    tmp199 = tl.load(in_ptr2 + (tmp72 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp200 = tl.load(in_ptr3 + (tmp72 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp201 = tl.load(in_ptr4 + (tmp72 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp202 = tl.where(tmp18, tmp200, tmp201)
    tmp203 = tl.where(tmp13, tmp199, tmp202)
    tmp204 = tl.load(in_ptr5 + (tmp72 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp205 = tmp203 + tmp204
    tmp206 = tmp205 * tmp85
    tmp207 = tl.load(in_ptr2 + (tmp90 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp208 = tl.load(in_ptr3 + (tmp90 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp209 = tl.load(in_ptr4 + (tmp90 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp210 = tl.where(tmp18, tmp208, tmp209)
    tmp211 = tl.where(tmp13, tmp207, tmp210)
    tmp212 = tl.load(in_ptr5 + (tmp90 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp213 = tmp211 + tmp212
    tmp214 = tmp213 * tmp105
    tmp215 = tmp49 + tmp68
    tmp216 = tmp215 + tmp86
    tmp217 = tmp216 + tmp106
    tmp218 = x1
    tmp219 = tmp218.to(tl.float32)
    tmp220 = tmp219 * tmp30
    tmp221 = libdevice.floor(tmp220)
    tmp222 = tmp220 - tmp221
    tmp223 = triton_helpers.maximum(tmp222, tmp34)
    tmp224 = triton_helpers.minimum(tmp223, tmp36)
    tmp225 = tmp224 + tmp36
    tmp226 = tmp225 * tmp39
    tmp227 = tmp226 - tmp41
    tmp228 = tmp227 * tmp225
    tmp229 = tmp228 + tmp44
    tmp230 = tmp229 * tmp225
    tmp231 = tmp230 - tmp47
    tmp232 = tmp217 * tmp231
    tmp233 = tmp118 + tmp126
    tmp234 = tmp233 + tmp134
    tmp235 = tmp234 + tmp142
    tmp236 = tmp224 * tmp61
    tmp237 = tmp236 - tmp63
    tmp238 = tmp237 * tmp224
    tmp239 = tmp238 * tmp224
    tmp240 = tmp239 + tmp36
    tmp241 = tmp235 * tmp240
    tmp242 = tmp232 + tmp241
    tmp243 = tmp154 + tmp162
    tmp244 = tmp243 + tmp170
    tmp245 = tmp244 + tmp178
    tmp246 = tmp36 - tmp224
    tmp247 = tmp246 * tmp61
    tmp248 = tmp247 - tmp63
    tmp249 = tmp248 * tmp246
    tmp250 = tmp249 * tmp246
    tmp251 = tmp250 + tmp36
    tmp252 = tmp245 * tmp251
    tmp253 = tmp242 + tmp252
    tmp254 = tmp190 + tmp198
    tmp255 = tmp254 + tmp206
    tmp256 = tmp255 + tmp214
    tmp257 = tmp98 - tmp224
    tmp258 = tmp257 * tmp39
    tmp259 = tmp258 - tmp41
    tmp260 = tmp259 * tmp257
    tmp261 = tmp260 + tmp44
    tmp262 = tmp261 * tmp257
    tmp263 = tmp262 - tmp47
    tmp264 = tmp256 * tmp263
    tmp265 = tmp253 + tmp264
    tmp266 = tl.load(in_ptr12 + (x5 + 16*(x2) + 32*x3), tmp13 & xmask, other=0.0)
    tmp267 = tl.load(in_ptr13 + (x5 + 16*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp268 = tl.load(in_ptr14 + (x5 + 16*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp269 = tl.where(tmp18, tmp267, tmp268)
    tmp270 = tl.where(tmp13, tmp266, tmp269)
    tmp272 = tmp270 + tmp271
    tmp273 = tmp272 + tmp265
    tl.store(in_out_ptr0 + (x7), tmp273, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (2, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_7, (2, ), (1, ))
    assert_size_stride(primals_8, (2, ), (1, ))
    assert_size_stride(primals_9, (2, ), (1, ))
    assert_size_stride(primals_10, (2, ), (1, ))
    assert_size_stride(primals_11, (1, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_12, (1, ), (1, ))
    assert_size_stride(primals_13, (1, ), (1, ))
    assert_size_stride(primals_14, (1, ), (1, ))
    assert_size_stride(primals_15, (1, ), (1, ))
    assert_size_stride(primals_16, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (2, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_22, (2, ), (1, ))
    assert_size_stride(primals_23, (2, ), (1, ))
    assert_size_stride(primals_24, (2, ), (1, ))
    assert_size_stride(primals_25, (2, ), (1, ))
    assert_size_stride(primals_26, (1, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_27, (1, ), (1, ))
    assert_size_stride(primals_28, (1, ), (1, ))
    assert_size_stride(primals_29, (1, ), (1, ))
    assert_size_stride(primals_30, (1, ), (1, ))
    assert_size_stride(primals_31, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (4, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, ), (1, ))
    assert_size_stride(primals_36, (2, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_37, (2, ), (1, ))
    assert_size_stride(primals_38, (2, ), (1, ))
    assert_size_stride(primals_39, (2, ), (1, ))
    assert_size_stride(primals_40, (2, ), (1, ))
    assert_size_stride(primals_41, (1, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_42, (1, ), (1, ))
    assert_size_stride(primals_43, (1, ), (1, ))
    assert_size_stride(primals_44, (1, ), (1, ))
    assert_size_stride(primals_45, (1, ), (1, ))
    assert_size_stride(primals_46, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (4, ), (1, ))
    assert_size_stride(primals_49, (4, ), (1, ))
    assert_size_stride(primals_50, (4, ), (1, ))
    assert_size_stride(primals_51, (2, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_52, (2, ), (1, ))
    assert_size_stride(primals_53, (2, ), (1, ))
    assert_size_stride(primals_54, (2, ), (1, ))
    assert_size_stride(primals_55, (2, ), (1, ))
    assert_size_stride(primals_56, (1, 2, 3, 3), (18, 9, 3, 1))
    assert_size_stride(primals_57, (1, ), (1, ))
    assert_size_stride(primals_58, (1, ), (1, ))
    assert_size_stride(primals_59, (1, ), (1, ))
    assert_size_stride(primals_60, (1, ), (1, ))
    assert_size_stride(primals_61, (1, 1, 3, 3), (9, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1, out1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(primals_1, primals_2, primals_3, primals_4, primals_5, buf0, 256, grid=grid(256), stream=stream0)
        del primals_4
        del primals_5
        # Topologically Sorted Source Nodes: [out1_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 2, 4, 4), (32, 16, 4, 1))
        buf2 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2, out2_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf1, primals_7, primals_8, primals_9, primals_10, buf2, 128, grid=grid(128), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [out2_2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 1, 4, 4), (16, 16, 4, 1))
        buf4 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3, out3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, primals_12, primals_13, primals_14, primals_15, buf4, 64, grid=grid(64), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [out3_2], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 1, 4, 4), (16, 16, 4, 1))
        buf6 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf7 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [low1, out1_3, out1_4], Original ATen: [aten.avg_pool2d, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_relu_3.run(primals_1, primals_17, primals_18, primals_19, primals_20, buf6, buf7, 64, grid=grid(64), stream=stream0)
        del primals_19
        del primals_20
        # Topologically Sorted Source Nodes: [out1_5], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 2, 2, 2), (8, 4, 2, 1))
        buf9 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_3, out2_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, primals_22, primals_23, primals_24, primals_25, buf9, 32, grid=grid(32), stream=stream0)
        del primals_25
        # Topologically Sorted Source Nodes: [out2_5], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 1, 2, 2), (4, 4, 2, 1))
        buf11 = empty_strided_cuda((4, 1, 2, 2), (4, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_5, out3_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf10, primals_27, primals_28, primals_29, primals_30, buf11, 16, grid=grid(16), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [out3_7], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 1, 2, 2), (4, 4, 2, 1))
        buf13 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf14 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_8, out3_9, out1_6, out1_7], Original ATen: [aten.cat, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_relu_6.run(buf8, buf10, buf12, buf6, primals_32, primals_33, primals_34, primals_35, buf13, buf14, 64, grid=grid(64), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [out1_8], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 2, 2, 2), (8, 4, 2, 1))
        buf16 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_6, out2_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf15, primals_37, primals_38, primals_39, primals_40, buf16, 32, grid=grid(32), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [out2_8], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 1, 2, 2), (4, 4, 2, 1))
        buf18 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [out3_10, out3_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf17, primals_42, primals_43, primals_44, primals_45, buf18, 16, grid=grid(16), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [out3_12], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 1, 2, 2), (4, 4, 2, 1))
        buf20 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf21 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf54 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out3_9, out3_13, out3_14, out1_9, out1_10], Original ATen: [aten.add, aten.cat, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_cat_native_batch_norm_backward_relu_7.run(buf15, buf17, buf19, buf13, buf6, primals_47, primals_48, primals_49, primals_50, buf20, buf21, buf54, 64, grid=grid(64), stream=stream0)
        del primals_47
        del primals_50
        # Topologically Sorted Source Nodes: [out1_11], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 2, 2, 2), (8, 4, 2, 1))
        buf23 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out2_9, out2_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf22, primals_52, primals_53, primals_54, primals_55, buf23, 32, grid=grid(32), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [out2_11], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 1, 2, 2), (4, 4, 2, 1))
        buf25 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [out3_15, out3_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf24, primals_57, primals_58, primals_59, primals_60, buf25, 16, grid=grid(16), stream=stream0)
        del primals_60
        # Topologically Sorted Source Nodes: [out3_17], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 1, 2, 2), (4, 4, 2, 1))
        buf27 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_8.run(buf27, 4, grid=grid(4), stream=stream0)
        buf28 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_8.run(buf28, 4, grid=grid(4), stream=stream0)
        buf29 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_floor_mul_9.run(buf29, 4, grid=grid(4), stream=stream0)
        buf30 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_10.run(buf30, 4, grid=grid(4), stream=stream0)
        buf31 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.floor, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_11.run(buf31, 4, grid=grid(4), stream=stream0)
        buf36 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_floor_mul_9.run(buf36, 4, grid=grid(4), stream=stream0)
        buf41 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_10.run(buf41, 4, grid=grid(4), stream=stream0)
        buf46 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [up2], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_11.run(buf46, 4, grid=grid(4), stream=stream0)
        buf32 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf51 = buf32; del buf32  # reuse
        buf52 = buf51; del buf51  # reuse
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [out3_3, out3_4, out3_18, out3_19, up2, add], Original ATen: [aten.cat, aten.add, aten.arange, aten._to_copy, aten.mul, aten.floor, aten.sub, aten.clamp, aten.rsub, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_rsub_sub_12.run(buf53, buf27, buf28, buf22, buf24, buf26, buf20, buf29, buf30, buf31, buf36, buf41, buf46, buf1, buf3, buf5, primals_1, 256, grid=grid(256), stream=stream0)
        del buf20
        del buf26
        del buf5
    return (buf53, primals_1, primals_2, primals_3, primals_6, primals_7, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_21, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_46, primals_48, primals_49, primals_51, primals_52, primals_53, primals_54, primals_56, primals_57, primals_58, primals_59, primals_61, buf0, buf1, buf2, buf3, buf4, buf6, buf7, buf8, buf9, buf10, buf11, buf13, buf14, buf15, buf16, buf17, buf18, buf21, buf22, buf23, buf24, buf25, buf27, buf28, buf29, buf30, buf31, buf36, buf41, buf46, buf54, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((2, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((2, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1, 2, 3, 3), (18, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
