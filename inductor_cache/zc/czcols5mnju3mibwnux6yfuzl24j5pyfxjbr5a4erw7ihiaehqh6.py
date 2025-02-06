# AOT ID: ['57_forward']
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


# kernel path: inductor_cache/jj/cjjn56ovnqf5bx4o6c2cb3vh2in2i4mhueoqdy3xryplbdxx3obb.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_1), kwargs = {})
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
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/uj/cujd54464orfiqj52ga3wwbv7lxm5of4imwkyjdrzwcjosgmotdy.py
# Topologically Sorted Source Nodes: [sim_map_2], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   sim_map_2 => div, exp, sum_1
# Graph fragment:
#   %mul_tensor_28 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm, 1), kwargs = {})
#   %amax_default_14 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_28, [-1], True), kwargs = {})
#   %sub_tensor_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_28, %amax_default_14), kwargs = {})
#   %mul_tensor_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_14, 0.7071067811865476), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_29,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_1 = async_compile.triton('triton_per_fused__softmax_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_1(in_out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = 0.7071067811865476
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(in_out_ptr0 + (r1 + 16*x0), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kv/ckvxx7u5lrcx255edizcrffd3we2ekcltx4ofuyjqxgsc3fqdzpe.py
# Topologically Sorted Source Nodes: [value], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   value => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/el/celgtg6k3bq66k5xrditrl2mr4ynpxbydgsnjv22vrwqtp67ho36.py
# Topologically Sorted Source Nodes: [context_local_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   context_local_1 => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 64*y1), xmask & ymask)
    tl.store(out_ptr0 + (x2 + 16*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/n7/cn73y5xv2qitvgkhy4bgzydh4qu2jqsuloqdeevq3chifrouv5dc.py
# Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_5 => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_26,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x1 + 16*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dq/cdqi7uubjcxtgun3ktgxqqhnkuqj4zzqzxzilqnthiubqzsg7jc3.py
# Topologically Sorted Source Nodes: [sim_map_5], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   sim_map_5 => exp_1
# Graph fragment:
#   %mul_tensor_26 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_2, 1), kwargs = {})
#   %amax_default_13 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_26, [-1], True), kwargs = {})
#   %sub_tensor_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_26, %amax_default_13), kwargs = {})
#   %mul_tensor_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_13, 0.7071067811865476), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_27,), kwargs = {})
triton_poi_fused__softmax_5 = async_compile.triton('triton_poi_fused__softmax_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 0.7071067811865476
    tmp16 = tmp14 * tmp15
    tmp17 = tl_math.exp(tmp16)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7v/c7vi5t63y4gj47yl7qhlrvwyqoxrtfkylfgahrifg4rsgqykt57s.py
# Topologically Sorted Source Nodes: [sim_map_5], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   sim_map_5 => div_1, sum_2
# Graph fragment:
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_2), kwargs = {})
triton_poi_fused__softmax_6 = async_compile.triton('triton_poi_fused__softmax_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uy/cuyb23szasv7taw5vewzznfstwk3gayq3sqw7yvx7vcruxq6yy7k.py
# Topologically Sorted Source Nodes: [contiguous_4, contiguous_8, contiguous_12, contiguous_16], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_12 => clone_11
#   contiguous_16 => clone_15
#   contiguous_4 => clone_3
#   contiguous_8 => clone_7
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_14,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_34,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_54,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_74,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x4 = xindex // 4
    x2 = ((xindex // 4) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x1 + 16*x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 4*x1 + 16*x4), xmask)
    tmp5 = tl.load(in_ptr0 + (8 + x0 + 4*x1 + 16*x4), xmask)
    tmp7 = tl.load(in_ptr0 + (10 + x0 + 4*x1 + 16*x4), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp7 + tmp1
    tl.store(out_ptr0 + (x5), tmp2, xmask)
    tl.store(out_ptr1 + (x5), tmp4, xmask)
    tl.store(out_ptr2 + (x5), tmp6, xmask)
    tl.store(out_ptr3 + (x5), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2w/c2w3pwtu3kqsfffd36fkkfqnvw4thhwb2hin4lhjze45cmjpuyzq.py
# Topologically Sorted Source Nodes: [contiguous_9], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_9 => clone_8
# Graph fragment:
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_46,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + x0 + 4*x1 + 16*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k4/ck4wiinv5t35rwwi72ghd4hjrtdil3ksddq6pnmsqc5gwwhu5bqw.py
# Topologically Sorted Source Nodes: [contiguous_13], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_13 => clone_12
# Graph fragment:
#   %clone_12 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_66,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + x0 + 4*x1 + 16*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ep/cep5eszhkbkonzntum4sle7bd2w2kmwaolyl35b3ibb32xnn2kxs.py
# Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_17 => clone_16
# Graph fragment:
#   %clone_16 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_86,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x2 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (10 + x0 + 4*x1 + 16*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nv/cnvq3vuh2gw6soyd5wxfztvlnxchktemh57u5hr67ozxn2tshavv.py
# Topologically Sorted Source Nodes: [context_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   context_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat, %cat_1], 2), kwargs = {})
triton_poi_fused_cat_11 = async_compile.triton('triton_poi_fused_cat_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x2 + 4*(x0) + 8*(x1) + 16*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 4, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp4
    tmp16 = tl.load(in_ptr1 + (x2 + 4*((-2) + x0) + 8*(x1) + 16*x3), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp9, tmp11, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 4, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = x0
    tmp24 = tl.full([1], 0, tl.int64)
    tmp25 = tmp23 >= tmp24
    tmp26 = tl.full([1], 2, tl.int64)
    tmp27 = tmp23 < tmp26
    tmp28 = tmp27 & tmp20
    tmp29 = tl.load(in_ptr2 + (x2 + 4*(x0) + 8*((-2) + x1) + 16*x3), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp23 >= tmp26
    tmp31 = tl.full([1], 4, tl.int64)
    tmp32 = tmp23 < tmp31
    tmp33 = tmp30 & tmp20
    tmp34 = tl.load(in_ptr3 + (x2 + 4*((-2) + x0) + 8*((-2) + x1) + 16*x3), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp27, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp20, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp19, tmp37)
    tl.store(out_ptr0 + (x4), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qv/cqvvcn5kvarihcdxlzshk3mwv4t3q43tq6sywfy2upeofmef5km4.py
# Topologically Sorted Source Nodes: [contiguous_21], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_21 => clone_20
# Graph fragment:
#   %clone_20 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_106,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aa/caahts4snxeirqbhhrcnvfhoohpt4ekfw7oomvc4gmkw4o6lqsuq.py
# Topologically Sorted Source Nodes: [sim_map_17], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   sim_map_17 => div_5, exp_5, sum_6
# Graph fragment:
#   %mul_tensor_18 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_10, 1), kwargs = {})
#   %amax_default_9 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_18, [-1], True), kwargs = {})
#   %sub_tensor_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_18, %amax_default_9), kwargs = {})
#   %mul_tensor_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_9, 0.7071067811865476), kwargs = {})
#   %exp_5 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_19,), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_5, [-1], True), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_5, %sum_6), kwargs = {})
triton_poi_fused__softmax_13 = async_compile.triton('triton_poi_fused__softmax_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_13(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2 - tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp7 = tmp6 / tmp6
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ek/cektxqvdx7jwxvwbpmk23krrv4jhwtpyodh22ozjcaiqajv2xoax.py
# Topologically Sorted Source Nodes: [contiguous_20, contiguous_24, contiguous_32, contiguous_36], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_20 => clone_19
#   contiguous_24 => clone_22
#   contiguous_32 => clone_29
#   contiguous_36 => clone_32
# Graph fragment:
#   %clone_19 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_94,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_22 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_114,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_29 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_154,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_32 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_174,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_14 = async_compile.triton('triton_poi_fused_clone_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_14(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (5 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp7 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr1 + (x2), tmp4, xmask)
    tl.store(out_ptr2 + (x2), tmp6, xmask)
    tl.store(out_ptr3 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/an/cangy27vi347sfgpk3ux44vn6pwdijbhk6yyz6uvl2lituebkmko.py
# Topologically Sorted Source Nodes: [contiguous_25], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_25 => clone_23
# Graph fragment:
#   %clone_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_126,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_15 = async_compile.triton('triton_poi_fused_clone_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oj/coj6mk2qiestejm35utjlrso2grvtgpxcws4pschuzzn7fnrt6kz.py
# Topologically Sorted Source Nodes: [contiguous_29], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_29 => clone_26
# Graph fragment:
#   %clone_26 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_146,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_16 = async_compile.triton('triton_poi_fused_clone_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + x0 + 16*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ci/ccituzdgoitc22imgy7uoir45mtnuacw4mti5tvmavojagd7mu22.py
# Topologically Sorted Source Nodes: [sim_map_23], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   sim_map_23 => div_7, exp_7, sum_8
# Graph fragment:
#   %mul_tensor_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%bmm_14, 1), kwargs = {})
#   %amax_default_7 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_14, [-1], True), kwargs = {})
#   %sub_tensor_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_14, %amax_default_7), kwargs = {})
#   %mul_tensor_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_7, 0.7071067811865476), kwargs = {})
#   %exp_7 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_15,), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_7, [-1], True), kwargs = {})
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_7, %sum_8), kwargs = {})
triton_poi_fused__softmax_17 = async_compile.triton('triton_poi_fused__softmax_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (2*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 2*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp8 = tmp2 - tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp8 * tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tmp4 - tmp7
    tmp13 = tmp12 * tmp9
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp6 - tmp7
    tmp16 = tmp15 * tmp9
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp14 + tmp17
    tmp19 = tmp11 / tmp18
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cr/ccrxfrd5ntvzq4dv6ltqydhfmlmb7jjp4wef233dqe4jfc7ewskb.py
# Topologically Sorted Source Nodes: [contiguous_28, contiguous_40, contiguous_44, contiguous_48], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_28 => clone_25
#   contiguous_40 => clone_35
#   contiguous_44 => clone_39
#   contiguous_48 => clone_43
# Graph fragment:
#   %clone_25 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_134,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_35 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_194,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_39 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_214,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_43 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_234,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_18 = async_compile.triton('triton_poi_fused_clone_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_18(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x3 = xindex // 2
    x1 = ((xindex // 2) % 4)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + x0 + 16*x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (6 + x0 + 16*x3), xmask)
    tmp5 = tl.load(in_ptr0 + (8 + 4*x0 + 16*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (9 + 4*x0 + 16*x3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp1
    tmp6 = tmp5 + tmp1
    tmp8 = tmp7 + tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp4, xmask)
    tl.store(out_ptr2 + (x4), tmp6, xmask)
    tl.store(out_ptr3 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/la/clayrzlyaqjxfisbljisydyek4dmdjvgbsnvayctk6omhqz2dajr.py
# Topologically Sorted Source Nodes: [contiguous_33], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_33 => clone_30
# Graph fragment:
#   %clone_30 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_166,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_19 = async_compile.triton('triton_poi_fused_clone_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/be/cbellw5d4lzgjsm5ikm2uwcbthc6xkl4jl4jbvlge2s77uvyxfio.py
# Topologically Sorted Source Nodes: [contiguous_37], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_37 => clone_33
# Graph fragment:
#   %clone_33 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_186,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vq/cvqyx3budv5jlayp6ujqxwhjbqm5efzqidinyt3ktnqwp7udtbre.py
# Topologically Sorted Source Nodes: [contiguous_41], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_41 => clone_36
# Graph fragment:
#   %clone_36 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_206,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_21 = async_compile.triton('triton_poi_fused_clone_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (6 + x0 + 16*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ce/ccen3rsfqevjfz3gpjccfbe7zoshirnlmuyonpsszqqon555pryp.py
# Topologically Sorted Source Nodes: [contiguous_45], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_45 => clone_40
# Graph fragment:
#   %clone_40 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_226,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_22 = async_compile.triton('triton_poi_fused_clone_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + 4*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ig/cigmxb2ixylz7m7ov2ffa6hruruasf2wn5ydax2bwpx5h54flaem.py
# Topologically Sorted Source Nodes: [contiguous_49], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_49 => clone_44
# Graph fragment:
#   %clone_44 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_246,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_23 = async_compile.triton('triton_poi_fused_clone_23', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = xindex // 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (9 + 4*x0 + 16*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jl/cjlodb7sc63btnf7pfxp2vlb2366o67k3prau6dhqgao6c4xnk2k.py
# Topologically Sorted Source Nodes: [contiguous_52], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_52 => clone_47
# Graph fragment:
#   %clone_47 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_254,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_24 = async_compile.triton('triton_poi_fused_clone_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_24(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x4 = xindex // 4
    x2 = ((xindex // 4) % 4)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (10 + x0 + 4*x1 + 16*x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x5), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ud/cudva62zp6fn5eavyiq64wuehy6bc6fuqd6e4rpg24vr43cbqdi2.py
# Topologically Sorted Source Nodes: [context_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   context_4 => cat_6
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_3, %cat_4, %cat_5], 2), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex // 16
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x6 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x5), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp4
    tmp17 = tl.load(in_ptr1 + (x5), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp5 >= tmp13
    tmp19 = tl.full([1], 4, tl.int64)
    tmp20 = tmp5 < tmp19
    tmp21 = tmp18 & tmp4
    tmp22 = tl.load(in_ptr2 + (x2 + 4*((-2) + x0) + 8*x3), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tl.where(tmp15, tmp17, tmp22)
    tmp24 = tl.where(tmp9, tmp11, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 2, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = x0
    tmp32 = tl.full([1], 0, tl.int64)
    tmp33 = tmp31 >= tmp32
    tmp34 = tl.full([1], 1, tl.int64)
    tmp35 = tmp31 < tmp34
    tmp36 = tmp35 & tmp30
    tmp37 = tl.load(in_ptr3 + (x5), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp31 >= tmp34
    tmp39 = tl.full([1], 2, tl.int64)
    tmp40 = tmp31 < tmp39
    tmp41 = tmp38 & tmp40
    tmp42 = tmp41 & tmp30
    tmp43 = tl.load(in_ptr4 + (x5), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp31 >= tmp39
    tmp45 = tl.full([1], 4, tl.int64)
    tmp46 = tmp31 < tmp45
    tmp47 = tmp44 & tmp30
    tmp48 = tl.load(in_ptr5 + (x2 + 4*((-2) + x0) + 8*x3), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.where(tmp41, tmp43, tmp48)
    tmp50 = tl.where(tmp35, tmp37, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 4, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = x0
    tmp57 = tl.full([1], 0, tl.int64)
    tmp58 = tmp56 >= tmp57
    tmp59 = tl.full([1], 1, tl.int64)
    tmp60 = tmp56 < tmp59
    tmp61 = tmp60 & tmp53
    tmp62 = tl.load(in_ptr6 + (x2 + 4*((-2) + x1) + 8*x3), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tmp56 >= tmp59
    tmp64 = tl.full([1], 2, tl.int64)
    tmp65 = tmp56 < tmp64
    tmp66 = tmp63 & tmp65
    tmp67 = tmp66 & tmp53
    tmp68 = tl.load(in_ptr7 + (x2 + 4*((-2) + x1) + 8*x3), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp69 = tmp56 >= tmp64
    tmp70 = tl.full([1], 4, tl.int64)
    tmp71 = tmp56 < tmp70
    tmp72 = tmp69 & tmp53
    tmp73 = tl.load(in_ptr8 + (x2 + 4*((-2) + x0) + 8*((-2) + x1) + 16*x3), tmp72 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.where(tmp66, tmp68, tmp73)
    tmp75 = tl.where(tmp60, tmp62, tmp74)
    tmp76 = tl.full(tmp75.shape, 0.0, tmp75.dtype)
    tmp77 = tl.where(tmp53, tmp75, tmp76)
    tmp78 = tl.where(tmp30, tmp52, tmp77)
    tmp79 = tl.where(tmp4, tmp26, tmp78)
    tl.store(out_ptr0 + (x6), tmp79, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6b/c6bqs7qf2wy2p6fewtbh2arnxbbxkvqzt56d3b5qm6mv3bki2iff.py
# Topologically Sorted Source Nodes: [cat_16], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_16 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu_8, %convolution_3, %convolution_7, %convolution_11, %convolution_15], 1), kwargs = {})
triton_poi_fused_cat_26 = async_compile.triton('triton_poi_fused_cat_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 32)
    x0 = (xindex % 16)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 256*x2), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 20, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (x0 + 16*((-16) + x1) + 64*x2), tmp28 & xmask, other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-16) + x1), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp28, tmp31, tmp32)
    tmp34 = tmp0 >= tmp26
    tmp35 = tl.full([1], 24, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tmp34 & tmp36
    tmp38 = tl.load(in_ptr7 + (x0 + 16*((-20) + x1) + 64*x2), tmp37 & xmask, other=0.0)
    tmp39 = tl.load(in_ptr8 + ((-20) + x1), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tmp38 + tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tmp0 >= tmp35
    tmp44 = tl.full([1], 28, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr9 + (x0 + 16*((-24) + x1) + 64*x2), tmp46 & xmask, other=0.0)
    tmp48 = tl.load(in_ptr10 + ((-24) + x1), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp46, tmp49, tmp50)
    tmp52 = tmp0 >= tmp44
    tmp53 = tl.full([1], 32, tl.int64)
    tmp54 = tmp0 < tmp53
    tmp55 = tl.load(in_ptr11 + (x0 + 16*((-28) + x1) + 64*x2), tmp52 & xmask, other=0.0)
    tmp56 = tl.load(in_ptr12 + ((-28) + x1), tmp52 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 + tmp56
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp52, tmp57, tmp58)
    tmp60 = tl.where(tmp46, tmp51, tmp59)
    tmp61 = tl.where(tmp37, tmp42, tmp60)
    tmp62 = tl.where(tmp28, tmp33, tmp61)
    tmp63 = tl.where(tmp4, tmp24, tmp62)
    tl.store(out_ptr0 + (x3), tmp63, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/if/cif2mwwpb54tnl665mnzgsd34dis3acqzivvnxsycejycrj6vzwc.py
# Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_29 => add_19, mul_78, mul_79, sub_24
#   input_30 => relu_9
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_73), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_75), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_78, %unsqueeze_77), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_79, %unsqueeze_79), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_9, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3), tmp17, xmask)
    tl.store(out_ptr1 + (x3), tmp19, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_5, (2, ), (1, ))
    assert_size_stride(primals_6, (2, ), (1, ))
    assert_size_stride(primals_7, (2, ), (1, ))
    assert_size_stride(primals_8, (2, ), (1, ))
    assert_size_stride(primals_9, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_14, (2, ), (1, ))
    assert_size_stride(primals_15, (2, ), (1, ))
    assert_size_stride(primals_16, (2, ), (1, ))
    assert_size_stride(primals_17, (2, ), (1, ))
    assert_size_stride(primals_18, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_23, (2, ), (1, ))
    assert_size_stride(primals_24, (2, ), (1, ))
    assert_size_stride(primals_25, (2, ), (1, ))
    assert_size_stride(primals_26, (2, ), (1, ))
    assert_size_stride(primals_27, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (2, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_32, (2, ), (1, ))
    assert_size_stride(primals_33, (2, ), (1, ))
    assert_size_stride(primals_34, (2, ), (1, ))
    assert_size_stride(primals_35, (2, ), (1, ))
    assert_size_stride(primals_36, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_37, (4, ), (1, ))
    assert_size_stride(primals_38, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_39, (16, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (16, ), (1, ))
    assert_size_stride(primals_42, (16, ), (1, ))
    assert_size_stride(primals_43, (4, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_44, (4, ), (1, ))
    assert_size_stride(primals_45, (4, ), (1, ))
    assert_size_stride(primals_46, (4, ), (1, ))
    assert_size_stride(primals_47, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [value], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_1, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 2, 4, 4), (32, 16, 4, 1))
        buf2 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf1, primals_5, primals_6, primals_7, primals_8, buf2, 128, grid=grid(128), stream=stream0)
        del primals_8
        buf3 = empty_strided_cuda((4, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (4, 16, 2), (32, 1, 16), 0), reinterpret_tensor(buf2, (4, 2, 16), (32, 16, 1), 0), out=buf3)
        buf6 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [sim_map_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_1.run(buf6, 64, 16, grid=grid(64), stream=stream0)
        buf7 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [value], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf7, primals_3, 256, grid=grid(256), stream=stream0)
        del primals_3
        buf8 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf6, reinterpret_tensor(buf7, (4, 16, 4), (64, 1, 16), 0), out=buf8)
        buf9 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf8, buf9, 16, 16, grid=grid(16, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [context_1], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(reinterpret_tensor(buf9, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [value_1], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(primals_1, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(primals_1, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 2, 4, 4), (32, 16, 4, 1))
        buf13 = empty_strided_cuda((4, 2, 4, 4), (32, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf12, primals_14, primals_15, primals_16, primals_17, buf13, 128, grid=grid(128), stream=stream0)
        buf14 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf13, buf14, 32, grid=grid(32), stream=stream0)
        buf15 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf14, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf14, (4, 2, 4), (8, 4, 1), 0), out=buf15)
        buf16 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_5], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf15, buf16, 64, grid=grid(64), stream=stream0)
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [sim_map_5], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_6.run(buf16, buf17, 64, grid=grid(64), stream=stream0)
        buf18 = reinterpret_tensor(buf16, (4, 4, 2, 2), (16, 4, 2, 1), 0); del buf16  # reuse
        buf24 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf36 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_4, contiguous_8, contiguous_12, contiguous_16], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf11, primals_12, buf18, buf24, buf30, buf36, 64, grid=grid(64), stream=stream0)
        del primals_12
        buf19 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf17, reinterpret_tensor(buf18, (4, 4, 4), (16, 1, 4), 0), out=buf19)
        buf20 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf13, buf20, 32, grid=grid(32), stream=stream0)
        buf21 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf20, (4, 2, 4), (8, 4, 1), 0), out=buf21)
        buf22 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_8], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf21, buf22, 64, grid=grid(64), stream=stream0)
        buf23 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [sim_map_8], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_6.run(buf22, buf23, 64, grid=grid(64), stream=stream0)
        buf25 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [context_local_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf23, reinterpret_tensor(buf24, (4, 4, 4), (16, 1, 4), 0), out=buf25)
        buf26 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_13], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf13, buf26, 32, grid=grid(32), stream=stream0)
        buf27 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf26, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf26, (4, 2, 4), (8, 4, 1), 0), out=buf27)
        buf28 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_11], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf27, buf28, 64, grid=grid(64), stream=stream0)
        buf29 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [sim_map_11], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_6.run(buf28, buf29, 64, grid=grid(64), stream=stream0)
        buf31 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [context_local_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf29, reinterpret_tensor(buf30, (4, 4, 4), (16, 1, 4), 0), out=buf31)
        buf32 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_17], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_10.run(buf13, buf32, 32, grid=grid(32), stream=stream0)
        buf33 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf32, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf32, (4, 2, 4), (8, 4, 1), 0), out=buf33)
        buf34 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_14], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf33, buf34, 64, grid=grid(64), stream=stream0)
        buf35 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [sim_map_14], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_6.run(buf34, buf35, 64, grid=grid(64), stream=stream0)
        buf37 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [context_local_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf35, reinterpret_tensor(buf36, (4, 4, 4), (16, 1, 4), 0), out=buf37)
        buf38 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [context_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_11.run(buf19, buf25, buf31, buf37, buf38, 256, grid=grid(256), stream=stream0)
        del buf19
        # Topologically Sorted Source Nodes: [context_3], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [value_2], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(primals_1, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(primals_1, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 2, 4, 4), (32, 16, 4, 1))
        buf42 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf41, primals_23, primals_24, primals_25, primals_26, buf42, 128, grid=grid(128), stream=stream0)
        buf43 = empty_strided_cuda((4, 2, 1, 1), (2, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_21], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf42, buf43, 8, grid=grid(8), stream=stream0)
        buf44 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (4, 1, 2), (2, 0, 1), 0), reinterpret_tensor(buf43, (4, 2, 1), (2, 1, 0), 0), out=buf44)
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [sim_map_17], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf45, 4, grid=grid(4), stream=stream0)
        buf46 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf51 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf61 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf66 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_20, contiguous_24, contiguous_32, contiguous_36], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_14.run(buf40, primals_21, buf46, buf51, buf61, buf66, 16, grid=grid(16), stream=stream0)
        buf47 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf45, reinterpret_tensor(buf46, (4, 1, 4), (4, 0, 1), 0), out=buf47)
        buf48 = empty_strided_cuda((4, 2, 1, 1), (2, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_25], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf42, buf48, 8, grid=grid(8), stream=stream0)
        buf49 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (4, 1, 2), (2, 0, 1), 0), reinterpret_tensor(buf48, (4, 2, 1), (2, 1, 0), 0), out=buf49)
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [sim_map_20], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf50, 4, grid=grid(4), stream=stream0)
        buf52 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf50, reinterpret_tensor(buf51, (4, 1, 4), (4, 0, 1), 0), out=buf52)
        buf53 = empty_strided_cuda((4, 2, 1, 2), (4, 2, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_29], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_16.run(buf42, buf53, 16, grid=grid(16), stream=stream0)
        buf54 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf53, (4, 2, 2), (4, 2, 1), 0), out=buf54)
        buf55 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_23], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_17.run(buf54, buf55, 16, grid=grid(16), stream=stream0)
        buf56 = empty_strided_cuda((4, 4, 1, 2), (8, 2, 2, 1), torch.float32)
        buf71 = empty_strided_cuda((4, 4, 1, 2), (8, 2, 2, 1), torch.float32)
        buf76 = empty_strided_cuda((4, 4, 2, 1), (8, 2, 1, 1), torch.float32)
        buf81 = empty_strided_cuda((4, 4, 2, 1), (8, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_28, contiguous_40, contiguous_44, contiguous_48], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_18.run(buf40, primals_21, buf56, buf71, buf76, buf81, 32, grid=grid(32), stream=stream0)
        buf57 = empty_strided_cuda((4, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf55, reinterpret_tensor(buf56, (4, 2, 4), (8, 1, 2), 0), out=buf57)
        buf58 = empty_strided_cuda((4, 2, 1, 1), (2, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_33], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_19.run(buf42, buf58, 8, grid=grid(8), stream=stream0)
        buf59 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf58, (4, 1, 2), (2, 0, 1), 0), reinterpret_tensor(buf58, (4, 2, 1), (2, 1, 0), 0), out=buf59)
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [sim_map_26], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf60, 4, grid=grid(4), stream=stream0)
        buf62 = reinterpret_tensor(buf54, (4, 1, 4), (4, 4, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [context_local_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf60, reinterpret_tensor(buf61, (4, 1, 4), (4, 0, 1), 0), out=buf62)
        buf63 = empty_strided_cuda((4, 2, 1, 1), (2, 1, 8, 8), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_37], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_20.run(buf42, buf63, 8, grid=grid(8), stream=stream0)
        buf64 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf63, (4, 1, 2), (2, 0, 1), 0), reinterpret_tensor(buf63, (4, 2, 1), (2, 1, 0), 0), out=buf64)
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [sim_map_29], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_13.run(buf65, 4, grid=grid(4), stream=stream0)
        buf67 = empty_strided_cuda((4, 1, 4), (4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf65, reinterpret_tensor(buf66, (4, 1, 4), (4, 0, 1), 0), out=buf67)
        buf68 = empty_strided_cuda((4, 2, 1, 2), (4, 2, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_41], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_21.run(buf42, buf68, 16, grid=grid(16), stream=stream0)
        buf69 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf68, (4, 2, 2), (4, 2, 1), 0), out=buf69)
        buf70 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_32], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_17.run(buf69, buf70, 16, grid=grid(16), stream=stream0)
        buf72 = empty_strided_cuda((4, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf70, reinterpret_tensor(buf71, (4, 2, 4), (8, 1, 2), 0), out=buf72)
        buf73 = reinterpret_tensor(buf69, (4, 2, 2, 1), (4, 2, 1, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [contiguous_45], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_22.run(buf42, buf73, 16, grid=grid(16), stream=stream0)
        buf74 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf73, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf73, (4, 2, 2), (4, 2, 1), 0), out=buf74)
        buf75 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_35], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_17.run(buf74, buf75, 16, grid=grid(16), stream=stream0)
        buf77 = empty_strided_cuda((4, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf75, reinterpret_tensor(buf76, (4, 2, 4), (8, 1, 2), 0), out=buf77)
        buf78 = reinterpret_tensor(buf74, (4, 2, 2, 1), (4, 2, 1, 1), 0); del buf74  # reuse
        # Topologically Sorted Source Nodes: [contiguous_49], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_23.run(buf42, buf78, 16, grid=grid(16), stream=stream0)
        buf79 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf78, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf78, (4, 2, 2), (4, 2, 1), 0), out=buf79)
        buf80 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_38], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_17.run(buf79, buf80, 16, grid=grid(16), stream=stream0)
        del buf79
        buf82 = empty_strided_cuda((4, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf80, reinterpret_tensor(buf81, (4, 2, 4), (8, 1, 2), 0), out=buf82)
        buf83 = empty_strided_cuda((4, 2, 2, 2), (8, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_53], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_10.run(buf42, buf83, 32, grid=grid(32), stream=stream0)
        buf84 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [sim_map_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf83, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf83, (4, 2, 4), (8, 4, 1), 0), out=buf84)
        buf85 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [sim_map_41], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf84, buf85, 64, grid=grid(64), stream=stream0)
        buf86 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [sim_map_41], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_6.run(buf85, buf86, 64, grid=grid(64), stream=stream0)
        buf87 = reinterpret_tensor(buf85, (4, 4, 2, 2), (16, 4, 2, 1), 0); del buf85  # reuse
        # Topologically Sorted Source Nodes: [contiguous_52], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_24.run(buf40, primals_21, buf87, 64, grid=grid(64), stream=stream0)
        del primals_21
        buf88 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [context_local_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf86, reinterpret_tensor(buf87, (4, 4, 4), (16, 1, 4), 0), out=buf88)
        buf89 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [context_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf47, buf52, buf57, buf62, buf67, buf72, buf77, buf82, buf88, buf89, 256, grid=grid(256), stream=stream0)
        del buf47
        del buf52
        del buf57
        del buf62
        del buf67
        del buf72
        del buf77
        del buf82
        del buf88
        # Topologically Sorted Source Nodes: [context_5], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [value_3], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(primals_1, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(primals_1, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 2, 4, 4), (32, 16, 4, 1))
        buf93 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf92, primals_32, primals_33, primals_34, primals_35, buf93, 128, grid=grid(128), stream=stream0)
        del primals_35
        buf97 = empty_strided_cuda((4, 0, 0), (0, 0, 1), torch.float32)
        buf101 = empty_strided_cuda((4, 0, 0), (0, 0, 1), torch.float32)
        buf105 = empty_strided_cuda((4, 0, 0), (0, 0, 1), torch.float32)
        buf106 = empty_strided_cuda((4, 16, 16), (256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sim_map_147], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (4, 16, 2), (32, 1, 16), 0), reinterpret_tensor(buf93, (4, 2, 16), (32, 16, 1), 0), out=buf106)
        buf109 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [sim_map_149], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_1.run(buf109, 64, 16, grid=grid(64), stream=stream0)
        buf110 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [value_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf110, primals_30, 256, grid=grid(256), stream=stream0)
        del primals_30
        buf111 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [context_local_147], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf109, reinterpret_tensor(buf110, (4, 16, 4), (64, 1, 16), 0), out=buf111)
        buf112 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [context_local_148], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf111, buf112, 16, 16, grid=grid(16, 16), stream=stream0)
        del buf111
        # Topologically Sorted Source Nodes: [context_7], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(reinterpret_tensor(buf112, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(primals_1, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 16, 4, 4), (256, 16, 4, 1))
        buf115 = empty_strided_cuda((4, 32, 4, 4), (512, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_26.run(buf114, primals_39, primals_40, primals_41, primals_42, buf10, primals_10, buf39, primals_19, buf90, primals_28, buf113, primals_37, buf115, 2048, grid=grid(2048), stream=stream0)
        del buf10
        del buf113
        del buf39
        del primals_10
        del primals_19
        del primals_28
        del primals_37
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 4, 4, 4), (64, 16, 4, 1))
        buf117 = buf90; del buf90  # reuse
        buf118 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_27.run(buf116, primals_44, primals_45, primals_46, primals_47, buf117, buf118, 256, grid=grid(256), stream=stream0)
        del primals_47
        buf119 = empty_strided_cuda((4, 0, 0), (0, 0, 1), torch.float32)
        buf120 = empty_strided_cuda((4, 0, 0), (0, 0, 1), torch.float32)
        buf121 = empty_strided_cuda((4, 0, 0), (0, 0, 1), torch.float32)
        buf122 = empty_strided_cuda((4, 2, 0, 4), (0, 0, 4, 1), torch.float32)
        buf123 = empty_strided_cuda((4, 4, 0, 4), (0, 0, 4, 1), torch.float32)
    return (buf117, primals_1, primals_2, primals_4, primals_5, primals_6, primals_7, primals_9, primals_11, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_29, primals_31, primals_32, primals_33, primals_34, primals_36, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, buf1, buf2, buf6, reinterpret_tensor(buf9, (4, 4, 4, 4), (64, 16, 4, 1), 0), buf12, buf17, buf23, buf29, buf35, buf38, buf41, buf45, buf50, buf55, buf60, buf65, buf70, buf75, buf80, buf86, buf89, buf92, buf93, buf97, buf101, buf105, buf109, reinterpret_tensor(buf112, (4, 4, 4, 4), (64, 16, 4, 1), 0), buf114, buf115, buf116, buf118, reinterpret_tensor(buf110, (4, 4, 16), (64, 16, 1), 0), reinterpret_tensor(buf105, (4, 0, 0), (0, 1, 0), 0), reinterpret_tensor(buf110, (4, 4, 0), (0, 0, 0), 0), buf119, reinterpret_tensor(buf93, (4, 2, 0), (0, 0, 0), 0), reinterpret_tensor(buf93, (4, 0, 2), (0, 0, 0), 0), reinterpret_tensor(buf101, (4, 0, 0), (0, 1, 0), 0), reinterpret_tensor(buf110, (4, 4, 0), (0, 0, 0), 0), buf120, reinterpret_tensor(buf93, (4, 2, 0), (0, 0, 0), 0), reinterpret_tensor(buf93, (4, 0, 2), (0, 0, 0), 0), reinterpret_tensor(buf97, (4, 0, 0), (0, 1, 0), 0), reinterpret_tensor(buf110, (4, 4, 0), (0, 0, 0), 0), buf121, reinterpret_tensor(buf93, (4, 2, 0), (0, 0, 0), 0), reinterpret_tensor(buf93, (4, 0, 2), (0, 0, 0), 0), buf122, buf123, reinterpret_tensor(buf87, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf83, (4, 2, 4), (8, 4, 1), 0), reinterpret_tensor(buf83, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf81, (4, 4, 2), (8, 2, 1), 0), reinterpret_tensor(buf78, (4, 2, 2), (4, 2, 1), 0), reinterpret_tensor(buf78, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf76, (4, 4, 2), (8, 2, 1), 0), reinterpret_tensor(buf73, (4, 2, 2), (4, 2, 1), 0), reinterpret_tensor(buf73, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf71, (4, 4, 2), (8, 2, 1), 0), reinterpret_tensor(buf68, (4, 2, 2), (4, 2, 1), 0), reinterpret_tensor(buf68, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf66, (4, 4, 1), (4, 1, 1), 0), reinterpret_tensor(buf63, (4, 2, 1), (2, 1, 1), 0), reinterpret_tensor(buf63, (4, 1, 2), (2, 1, 1), 0), reinterpret_tensor(buf61, (4, 4, 1), (4, 1, 1), 0), reinterpret_tensor(buf58, (4, 2, 1), (2, 1, 1), 0), reinterpret_tensor(buf58, (4, 1, 2), (2, 1, 1), 0), reinterpret_tensor(buf56, (4, 4, 2), (8, 2, 1), 0), reinterpret_tensor(buf53, (4, 2, 2), (4, 2, 1), 0), reinterpret_tensor(buf53, (4, 2, 2), (4, 1, 2), 0), reinterpret_tensor(buf51, (4, 4, 1), (4, 1, 1), 0), reinterpret_tensor(buf48, (4, 2, 1), (2, 1, 1), 0), reinterpret_tensor(buf48, (4, 1, 2), (2, 1, 1), 0), reinterpret_tensor(buf46, (4, 4, 1), (4, 1, 1), 0), reinterpret_tensor(buf43, (4, 2, 1), (2, 1, 1), 0), reinterpret_tensor(buf43, (4, 1, 2), (2, 1, 1), 0), reinterpret_tensor(buf36, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf32, (4, 2, 4), (8, 4, 1), 0), reinterpret_tensor(buf32, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf30, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf26, (4, 2, 4), (8, 4, 1), 0), reinterpret_tensor(buf26, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf24, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf20, (4, 2, 4), (8, 4, 1), 0), reinterpret_tensor(buf20, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf18, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf14, (4, 2, 4), (8, 4, 1), 0), reinterpret_tensor(buf14, (4, 4, 2), (8, 1, 4), 0), reinterpret_tensor(buf7, (4, 4, 16), (64, 16, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((4, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
