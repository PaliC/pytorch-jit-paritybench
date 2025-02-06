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


# kernel path: inductor_cache/wq/cwqd35lp22ypefzbeqjf6br7j6ueonpex75467e7sqv4nm43nsau.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   input_1 => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%primals_1, [-1, -2], True), kwargs = {})
triton_red_fused_mean_0 = async_compile.triton('triton_red_fused_mean_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_0(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp4 = 4096.0
    tmp5 = tmp2 / tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/o7/co7umherztjjl4wixyu74qkhlq5aozyafblxjypboqy3d22sptwa.py
# Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_3 => add_1, mul_1, mul_2, sub
#   input_4 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
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


# kernel path: inductor_cache/jw/cjwwnnn5ujkblx6xzvm3sjnkeczsjbbqusx7ty5aq3x4zxe2d2ex.py
# Topologically Sorted Source Nodes: [b1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   b1 => convert_element_type_3
# Graph fragment:
#   %convert_element_type_3 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/44/c44p2b3xnmq2uoouxenp5rxuwjldqn64vsfbu7xdscxx6br2lhdk.py
# Topologically Sorted Source Nodes: [b1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   b1 => clamp_max_2, clamp_min, clamp_min_2, convert_element_type_2, iota, mul_3, sub_1
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 0.0), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_3, 0.0), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_5), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_1, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_3 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_3', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/67/c67yxmnc5xauwfzavsfpxvtntttsuxv6z6ic5yghjdcaduxbqt3l.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_8 => getitem
# Graph fragment:
#   %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_4 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/qn/cqnysr2xps7p6fomk5nbkwzxsbpqenyx6axlrb3bqbrwdp3un5ik.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_10, mul_12, mul_13, sub_7
#   input_11 => relu_2
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_19), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_21), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
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
    tl.store(out_ptr0 + (x0), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/zk/czkge3vuh6t27fjvrib46e2t4ovwgujhzwdq4ekmhaxtydkbikjc.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_12 => getitem_2, getitem_3
# Graph fragment:
#   %getitem_2 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 0), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_6 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_6(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck3c5kjtmtnxn2msucjrkqz4zwe2xvoyw7at3mn5g55nfin7ttti.py
# Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_14 => add_12, mul_15, mul_16, sub_8
#   input_15 => relu_3
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_27), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %unsqueeze_29), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %unsqueeze_31), kwargs = {})
#   %relu_3 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
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


# kernel path: inductor_cache/bp/cbpqf5psauv6nqtb7t4bcbzllmuf2yquxuljdu5kst3dssq3nya6.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   input_16 => getitem_4, getitem_5
# Graph fragment:
#   %getitem_4 : [num_users=2] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 0), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_8 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + 2*x0 + 32*x1), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp7 = tmp1 > tmp0
    tmp8 = tl.full([1], 1, tl.int8)
    tmp9 = tl.full([1], 0, tl.int8)
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp3 > tmp2
    tmp12 = tl.full([1], 2, tl.int8)
    tmp13 = tl.where(tmp11, tmp12, tmp10)
    tmp14 = tmp5 > tmp4
    tmp15 = tl.full([1], 3, tl.int8)
    tmp16 = tl.where(tmp14, tmp15, tmp13)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(out_ptr1 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iu/ciucdduwfdkr5flhibjfxc6vnnikdewtzd7cpanpibk3apzkve72.py
# Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_18 => add_14, mul_18, mul_19, sub_9
#   input_19 => relu_4
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_35), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %unsqueeze_37), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_14,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: inductor_cache/zj/czjvmuqqla5upopj6hwtasokiwsg7umxkmhtw4autakkayh5td4m.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x3 => convert_element_type_21
# Graph fragment:
#   %convert_element_type_21 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_10 = async_compile.triton('triton_poi_fused__to_copy_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_10(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck3ccomacz6trr5kpjn3cdp4d4trernul6463g3bxym6cd6elf5h.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x3 => add_21, clamp_max_4
# Graph fragment:
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_21, 1), kwargs = {})
#   %clamp_max_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_21, 7), kwargs = {})
triton_poi_fused_add_clamp_11 = async_compile.triton('triton_poi_fused_add_clamp_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_11(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 7, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ub/cubogpz2add2fb5rschetqkq37drwi7wsljyob3ujsy5gyyg2esv.py
# Topologically Sorted Source Nodes: [x3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   x3 => clamp_max_6, clamp_min_4, clamp_min_6, convert_element_type_20, iota_2, mul_29, sub_13
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_20 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, 0.4666666666666667), kwargs = {})
#   %clamp_min_4 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_29, 0.0), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_4, %convert_element_type_23), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_13, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_12 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_12(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6k/c6kqv7fjritz33kj5p5vkozgvqptv6yq5fna3b4tdilidizqn7tk.py
# Topologically Sorted Source Nodes: [input_24, input_25, x3, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_24 => add_18, mul_24, mul_25, sub_11
#   input_25 => relu_6
#   x => add_26
#   x3 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_23, add_24, add_25, mul_31, mul_32, mul_33, sub_14, sub_15, sub_17
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_51), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_24, %unsqueeze_53), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_18,), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_7, [None, None, %convert_element_type_21, %convert_element_type_23]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_7, [None, None, %convert_element_type_21, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_7, [None, None, %clamp_max_4, %convert_element_type_23]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_7, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %clamp_max_6), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_31), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %clamp_max_6), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_32), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %add_23), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %clamp_max_7), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %mul_33), kwargs = {})
#   %add_26 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %relu_6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (x4), xmask)
    tmp33 = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp36 = tl.load(in_ptr9 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp46 = tl.load(in_ptr10 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp49 = tl.load(in_ptr11 + (0))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 8*tmp4 + 64*x2), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 8*tmp22 + 64*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 8*tmp22 + 64*x2), xmask, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp35 = tmp32 - tmp34
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp35 * tmp44
    tmp48 = tmp45 * tmp47
    tmp51 = tmp48 + tmp50
    tmp52 = tl.full([1], 0, tl.int32)
    tmp53 = triton_helpers.maximum(tmp52, tmp51)
    tmp54 = tmp31 + tmp53
    tl.store(in_out_ptr0 + (x4), tmp54, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jj/cjjnv7y5wsrz4dnqgirexxj4wahnggvjw3umu56buy57hr3rbcvc.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_1 => convert_element_type_25
# Graph fragment:
#   %convert_element_type_25 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_14 = async_compile.triton('triton_poi_fused__to_copy_14', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4838709677419355
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sa/csa7ss6ga5h57jjz6qn7h5llayzlpucnwcbf2rym23au4vbrkijr.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_1 => add_27, clamp_max_8
# Graph fragment:
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_25, 1), kwargs = {})
#   %clamp_max_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_27, 15), kwargs = {})
triton_poi_fused_add_clamp_15 = async_compile.triton('triton_poi_fused_add_clamp_15', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4838709677419355
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 15, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ur/curch3u62upsu7yixdneklnrroxtoaoqjgkblj76v5qx52obmuna.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   x_1 => clamp_max_10, clamp_min_10, clamp_min_8, convert_element_type_24, iota_4, mul_34, sub_18
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_24 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_4, torch.float32), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_24, 0.4838709677419355), kwargs = {})
#   %clamp_min_8 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_34, 0.0), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_8, %convert_element_type_27), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_18, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_16 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_16', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_16(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.4838709677419355
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mh/cmhttt4f5rwzo7csehw76jozfi2z3viemduelasyoztipxtcz5mq.py
# Topologically Sorted Source Nodes: [input_21, input_22, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_21 => add_16, mul_21, mul_22, sub_10
#   input_22 => relu_5
#   x_1 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_29, add_30, add_31, mul_36, mul_37, mul_38, sub_19, sub_20, sub_22
#   x_2 => add_32
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_26, [None, None, %convert_element_type_25, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_26, [None, None, %convert_element_type_25, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_26, [None, None, %clamp_max_8, %convert_element_type_27]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_26, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %clamp_max_10), kwargs = {})
#   %add_29 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_36), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %clamp_max_10), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_37), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_30, %add_29), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %clamp_max_11), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %mul_38), kwargs = {})
#   %add_32 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %relu_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr7 + (x4), None)
    tmp33 = tl.load(in_ptr8 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp36 = tl.load(in_ptr9 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp46 = tl.load(in_ptr10 + (0))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp49 = tl.load(in_ptr11 + (0))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 16*tmp4 + 256*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 16*tmp22 + 256*x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 16*tmp22 + 256*x2), None, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tmp35 = tmp32 - tmp34
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.sqrt(tmp39)
    tmp41 = tl.full([1], 1, tl.int32)
    tmp42 = tmp41 / tmp40
    tmp43 = 1.0
    tmp44 = tmp42 * tmp43
    tmp45 = tmp35 * tmp44
    tmp48 = tmp45 * tmp47
    tmp51 = tmp48 + tmp50
    tmp52 = tl.full([1], 0, tl.int32)
    tmp53 = triton_helpers.maximum(tmp52, tmp51)
    tmp54 = tmp31 + tmp53
    tl.store(in_out_ptr0 + (x4), tmp54, None)
''', device_str='cuda')


# kernel path: inductor_cache/sn/csniczs627h2u3qj7jrzlpgxouplcsopdatosppdkesgogt3nkwy.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   x_3 => convert_element_type_29
# Graph fragment:
#   %convert_element_type_29 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
triton_poi_fused__to_copy_18 = async_compile.triton('triton_poi_fused__to_copy_18', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_18(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.49206349206349204
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jg/cjgiwtklm3srs6mhsidv4li3ipzbaxmas52b24z3hfwvycylypvm.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   x_3 => add_33, clamp_max_12
# Graph fragment:
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_29, 1), kwargs = {})
#   %clamp_max_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_33, 31), kwargs = {})
triton_poi_fused_add_clamp_19 = async_compile.triton('triton_poi_fused_add_clamp_19', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_19(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.49206349206349204
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 31, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qk/cqkuzpzfm5pacabssn6jg7px6u6wsznhgm4jad3x5znqdmr5suqu.py
# Topologically Sorted Source Nodes: [b1, x_3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   b1 => convert_element_type_2, iota
#   x_3 => clamp_max_14, clamp_min_12, clamp_min_14, mul_39, sub_23
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 0.49206349206349204), kwargs = {})
#   %clamp_min_12 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_39, 0.0), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_12, %convert_element_type_31), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_23, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_20 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_20', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_20(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.49206349206349204
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qq/cqqukpvcuk6n2d74ztihundaaw5toog5gbrvo6n34pvotppgdwsy.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   x_3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_35, add_36, add_37, mul_41, mul_42, mul_43, sub_24, sub_25, sub_27
# Graph fragment:
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_32, [None, None, %convert_element_type_29, %convert_element_type_31]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_32, [None, None, %convert_element_type_29, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_32, [None, None, %clamp_max_12, %convert_element_type_31]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_32, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %clamp_max_14), kwargs = {})
#   %add_35 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_41), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %clamp_max_14), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_42), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_36, %add_35), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %clamp_max_15), kwargs = {})
#   %add_37 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %mul_43), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_21 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (tmp13 + 32*tmp4 + 1024*x2), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (tmp8 + 32*tmp22 + 1024*x2), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (tmp13 + 32*tmp22 + 1024*x2), None, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tl.store(in_out_ptr0 + (x4), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/gw/cgw4idrf2cplrhx7chpguwedfrsuqnhkrd4776lw6ve5rmseyhpw.py
# Topologically Sorted Source Nodes: [b1, input_6, input_7, x_4, x_5], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   b1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_4, add_5, add_6, mul_5, mul_6, mul_7, sub_2, sub_3, sub_5
#   input_6 => add_8, mul_10, mul_9, sub_6
#   input_7 => relu_1
#   x_4 => mul_44
#   x_5 => add_38
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %convert_element_type_3, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %convert_element_type_3, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %clamp_max, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %clamp_max_2), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_5), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_6), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %add_4), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %clamp_max_3), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_7), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_11), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_13), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_8,), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_37, %relu_1), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %add_6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = xindex // 4096
    x5 = xindex
    x4 = xindex // 16384
    x6 = (xindex % 4096)
    x3 = ((xindex // 4096) % 4)
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x6 + 4096*x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr6 + (x5), None)
    tmp20 = tl.load(in_ptr7 + (x3), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x3), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x3), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr10 + (x3), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
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
    tmp21 = tmp19 - tmp20
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tl.full([1], 1, tl.int32)
    tmp27 = tmp26 / tmp25
    tmp28 = 1.0
    tmp29 = tmp27 * tmp28
    tmp30 = tmp21 * tmp29
    tmp32 = tmp30 * tmp31
    tmp34 = tmp32 + tmp33
    tmp35 = tl.full([1], 0, tl.int32)
    tmp36 = triton_helpers.maximum(tmp35, tmp34)
    tmp37 = tmp18 * tmp36
    tmp39 = tmp38 + tmp1
    tmp40 = tmp38 < 0
    tmp41 = tl.where(tmp40, tmp39, tmp38)
    tmp42 = tmp17 - tmp17
    tmp44 = tmp42 * tmp43
    tmp45 = tmp17 + tmp44
    tmp46 = tmp37 + tmp45
    tl.store(in_out_ptr0 + (x5), tmp46, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_2, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (1, 4, 7, 7), (196, 49, 7, 1))
    assert_size_stride(primals_13, (1, ), (1, ))
    assert_size_stride(primals_14, (1, ), (1, ))
    assert_size_stride(primals_15, (1, ), (1, ))
    assert_size_stride(primals_16, (1, ), (1, ))
    assert_size_stride(primals_17, (1, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_18, (1, ), (1, ))
    assert_size_stride(primals_19, (1, ), (1, ))
    assert_size_stride(primals_20, (1, ), (1, ))
    assert_size_stride(primals_21, (1, ), (1, ))
    assert_size_stride(primals_22, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (1, ), (1, ))
    assert_size_stride(primals_24, (1, ), (1, ))
    assert_size_stride(primals_25, (1, ), (1, ))
    assert_size_stride(primals_26, (1, ), (1, ))
    assert_size_stride(primals_27, (1, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_28, (1, ), (1, ))
    assert_size_stride(primals_29, (1, ), (1, ))
    assert_size_stride(primals_30, (1, ), (1, ))
    assert_size_stride(primals_31, (1, ), (1, ))
    assert_size_stride(primals_32, (1, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_33, (1, ), (1, ))
    assert_size_stride(primals_34, (1, ), (1, ))
    assert_size_stride(primals_35, (1, ), (1, ))
    assert_size_stride(primals_36, (1, ), (1, ))
    assert_size_stride(primals_37, (1, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_38, (1, ), (1, ))
    assert_size_stride(primals_39, (1, ), (1, ))
    assert_size_stride(primals_40, (1, ), (1, ))
    assert_size_stride(primals_41, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf1 = reinterpret_tensor(buf0, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_0.run(buf1, primals_1, 16, 4096, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 4, 1, 1), (4, 1, 1, 1))
        buf3 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf2, primals_3, primals_4, primals_5, primals_6, buf3, 16, grid=grid(16), stream=stream0)
        buf4 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [b1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf4, 64, grid=grid(64), stream=stream0)
        buf5 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [b1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf5, 64, grid=grid(64), stream=stream0)
        buf6 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [b1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf6, 64, grid=grid(64), stream=stream0)
        buf7 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [b1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(buf7, 64, grid=grid(64), stream=stream0)
        buf8 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [b1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_3.run(buf8, 64, grid=grid(64), stream=stream0)
        buf10 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [b1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_3.run(buf10, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(primals_1, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf12 = empty_strided_cuda((4, 4, 32, 32), (4096, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_4.run(primals_1, buf12, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_12, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 1, 32, 32), (1024, 1024, 32, 1))
        buf14 = empty_strided_cuda((4, 1, 32, 32), (1024, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf13, primals_13, primals_14, primals_15, primals_16, buf14, 4096, grid=grid(4096), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf14, primals_27, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 1, 32, 32), (1024, 1024, 32, 1))
        buf15 = empty_strided_cuda((4, 1, 16, 16), (256, 256, 16, 1), torch.float32)
        buf16 = empty_strided_cuda((4, 1, 16, 16), (256, 256, 16, 1), torch.int8)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_6.run(buf14, buf15, buf16, 1024, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf15, primals_17, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 1, 16, 16), (256, 256, 16, 1))
        buf18 = empty_strided_cuda((4, 1, 16, 16), (256, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf17, primals_18, primals_19, primals_20, primals_21, buf18, 1024, grid=grid(1024), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf18, primals_32, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 1, 16, 16), (256, 256, 16, 1))
        buf19 = empty_strided_cuda((4, 1, 8, 8), (64, 64, 8, 1), torch.float32)
        buf20 = empty_strided_cuda((4, 1, 8, 8), (64, 64, 8, 1), torch.int8)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_8.run(buf18, buf19, buf20, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf19, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 1, 8, 8), (64, 64, 8, 1))
        buf22 = empty_strided_cuda((4, 1, 8, 8), (64, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf21, primals_23, primals_24, primals_25, primals_26, buf22, 256, grid=grid(256), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf22, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 1, 8, 8), (64, 64, 8, 1))
        buf26 = empty_strided_cuda((4, 1, 8, 8), (64, 256, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf25, primals_38, primals_39, primals_40, primals_41, buf26, 256, grid=grid(256), stream=stream0)
        buf27 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(buf27, 16, grid=grid(16), stream=stream0)
        buf28 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_11.run(buf28, 16, grid=grid(16), stream=stream0)
        buf29 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_10.run(buf29, 16, grid=grid(16), stream=stream0)
        buf30 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_11.run(buf30, 16, grid=grid(16), stream=stream0)
        buf31 = empty_strided_cuda((16, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_12.run(buf31, 16, grid=grid(16), stream=stream0)
        buf33 = empty_strided_cuda((16, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_12.run(buf33, 16, grid=grid(16), stream=stream0)
        buf32 = empty_strided_cuda((4, 1, 16, 16), (256, 1024, 16, 1), torch.float32)
        buf34 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25, x3, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_13.run(buf34, buf27, buf29, buf26, buf30, buf31, buf28, buf33, buf24, primals_33, primals_34, primals_35, primals_36, 1024, grid=grid(1024), stream=stream0)
        del buf26
        buf35 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf35, 32, grid=grid(32), stream=stream0)
        buf36 = empty_strided_cuda((32, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_15.run(buf36, 32, grid=grid(32), stream=stream0)
        buf37 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_14.run(buf37, 32, grid=grid(32), stream=stream0)
        buf38 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_15.run(buf38, 32, grid=grid(32), stream=stream0)
        buf39 = empty_strided_cuda((32, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_16.run(buf39, 32, grid=grid(32), stream=stream0)
        buf41 = empty_strided_cuda((32, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_16.run(buf41, 32, grid=grid(32), stream=stream0)
        buf40 = empty_strided_cuda((4, 1, 32, 32), (1024, 4096, 32, 1), torch.float32)
        buf42 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_21, input_22, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_17.run(buf42, buf35, buf37, buf34, buf38, buf39, buf36, buf41, buf23, primals_28, primals_29, primals_30, primals_31, 4096, grid=grid(4096), stream=stream0)
        del buf34
        buf43 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_18.run(buf43, 64, grid=grid(64), stream=stream0)
        buf44 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_19.run(buf44, 64, grid=grid(64), stream=stream0)
        buf45 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [b1, x_3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_18.run(buf45, 64, grid=grid(64), stream=stream0)
        buf46 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_19.run(buf46, 64, grid=grid(64), stream=stream0)
        buf47 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [b1, x_3], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_20.run(buf47, 64, grid=grid(64), stream=stream0)
        buf49 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_20.run(buf49, 64, grid=grid(64), stream=stream0)
        buf48 = empty_strided_cuda((4, 1, 64, 64), (4096, 16384, 64, 1), torch.float32)
        buf50 = reinterpret_tensor(buf48, (4, 1, 64, 64), (4096, 4096, 64, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_21.run(buf50, buf43, buf45, buf42, buf46, buf47, buf44, buf49, 16384, grid=grid(16384), stream=stream0)
        del buf42
        buf9 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        buf51 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [b1, input_6, input_7, x_4, x_5], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_22.run(buf51, buf4, buf6, buf3, buf7, buf8, buf50, buf11, primals_8, primals_9, primals_10, primals_11, buf5, buf10, 65536, grid=grid(65536), stream=stream0)
        del buf3
    return (buf51, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, buf1, buf2, buf4, buf5, buf6, buf7, buf8, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf27, buf28, buf29, buf30, buf31, buf33, buf35, buf36, buf37, buf38, buf39, buf41, buf43, buf44, buf45, buf46, buf47, buf49, buf50, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 4, 7, 7), (196, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, 1, 7, 7), (49, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
