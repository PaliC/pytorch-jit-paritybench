# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/cl/cclz2vz4kwgnlnm32x532imbwxut4damzffqtcgsawi3fz2yud2y.py
# Topologically Sorted Source Nodes: [pad], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   pad => _unsafe_index, _unsafe_index_1
# Graph fragment:
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_3, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_0 = async_compile.triton('triton_poi_fused_reflection_pad2d_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 62208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 72)
    x1 = ((xindex // 72) % 72)
    x2 = xindex // 5184
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-4) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-4) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ib/cibqgvawnhfdmtgvyuppimhbpptsvhouqrhjtbqfqvrad2jgmfd2.py
# Topologically Sorted Source Nodes: [conv2d, instance_norm], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   conv2d => convolution
#   instance_norm => add, rsqrt, var_mean
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_red_fused__native_batch_norm_legit_convolution_1 = async_compile.triton('triton_red_fused__native_batch_norm_legit_convolution_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_convolution_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_convolution_1(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = (xindex % 32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r2 + 4096*x3), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
    tmp7 = 4096.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lc/clckb6daabu3cibk6fbecflhhukrbt6545g3jo4ozvnf3n2do5x7.py
# Topologically Sorted Source Nodes: [instance_norm], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   instance_norm => repeat
# Graph fragment:
#   %repeat : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_4, [4]), kwargs = {})
triton_poi_fused_repeat_2 = async_compile.triton('triton_poi_fused_repeat_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((x0 % 32)), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n4/cn4koykwzxk2zxyp2uuxzvvqbprqvo6bshktjowzymj3uq2oiagm.py
# Topologically Sorted Source Nodes: [y, pad_1], Original ATen: [aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   pad_1 => _unsafe_index_2, _unsafe_index_3
#   y => relu
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %sub_6, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_6]), kwargs = {})
triton_poi_fused_reflection_pad2d_relu_3 = async_compile.triton('triton_poi_fused_reflection_pad2d_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 557568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x2 = xindex // 4356
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eg/ceglcck5dnvvf7yfxcaqfpm43n7ajmrzjdxkaflqje3osj3eyqig.py
# Topologically Sorted Source Nodes: [conv2d_1, instance_norm_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
#   instance_norm_1 => add_2, rsqrt_1, var_mean_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %primals_6, %primals_7, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_4 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_4(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 256
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 1024*x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 1024, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 1024.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(in_out_ptr0 + (r2 + 1024*x3), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, None)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/mf/cmfqkedau3whtgjflktdtp7nxio6sa6gbbi7wfpbwf2zwvygjhbp.py
# Topologically Sorted Source Nodes: [instance_norm_1], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   instance_norm_1 => repeat_2
# Graph fragment:
#   %repeat_2 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_8, [4]), kwargs = {})
triton_poi_fused_repeat_5 = async_compile.triton('triton_poi_fused_repeat_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((x0 % 64)), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j3/cj3jtpte2avlvpemkaan4dpspyg5r7kh5lswozsqcqhqbvmjboo5.py
# Topologically Sorted Source Nodes: [y_1, pad_2], Original ATen: [aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   pad_2 => _unsafe_index_4, _unsafe_index_5
#   y_1 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_3,), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_1, [None, None, %sub_11, None]), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_4, [None, None, None, %sub_11]), kwargs = {})
triton_poi_fused_reflection_pad2d_relu_6 = async_compile.triton('triton_poi_fused_reflection_pad2d_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 295936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 34)
    x1 = ((xindex // 34) % 34)
    x2 = xindex // 1156
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1023 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0))) + ((-32)*tl_math.abs((-31) + tl_math.abs((-1) + x1))) + 1024*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6u/c6uekcyvfkzogcnaxlg2s5jejjlva6yhe2mc5x43g2x5u7cl6ibe.py
# Topologically Sorted Source Nodes: [conv2d_2, instance_norm_2, y_2], Original ATen: [aten.convolution, aten.repeat, aten._native_batch_norm_legit, aten.relu]
# Source node to ATen node mapping:
#   conv2d_2 => convolution_2
#   instance_norm_2 => add_4, repeat_4, repeat_5, rsqrt_2, var_mean_2
#   y_2 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_5, %primals_10, %primals_11, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %repeat_4 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_12, [4]), kwargs = {})
#   %repeat_5 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_13, [4]), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_5,), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_relu_repeat_7 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_relu_repeat_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_relu_repeat_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_relu_repeat_7(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel):
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
    x0 = xindex
    r3 = rindex
    x1 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + ((x0 % 128)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((x0 % 128)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_out_ptr0 + (r3 + 256*x0), None)
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 256, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 256.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp4 - tmp12
    tmp24 = tmp23 * tmp22
    tmp25 = tmp24 * tmp0
    tmp26 = tmp25 + tmp1
    tmp27 = tl.full([1], 0, tl.int32)
    tmp28 = triton_helpers.maximum(tmp27, tmp26)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp1, None)
    tl.store(in_out_ptr0 + (r3 + 256*x0), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp22, None)
    tl.store(out_ptr3 + (r3 + 256*x0), tmp28, None)
    tl.store(out_ptr2 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/ju/cjugqavvdnukf6ckdvfomfvt7hldlnymumlfttag4qdjxvjwgw4c.py
# Topologically Sorted Source Nodes: [pad_3], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   pad_3 => _unsafe_index_6, _unsafe_index_7
# Graph fragment:
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_2, [None, None, %sub_16, None]), kwargs = {})
#   %_unsafe_index_7 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_6, [None, None, None, %sub_16]), kwargs = {})
triton_poi_fused_reflection_pad2d_8 = async_compile.triton('triton_poi_fused_reflection_pad2d_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 165888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18)
    x1 = ((xindex // 18) % 18)
    x2 = xindex // 324
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (255 + ((-1)*tl_math.abs((-15) + tl_math.abs((-1) + x0))) + ((-16)*tl_math.abs((-15) + tl_math.abs((-1) + x1))) + 256*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6b/c6bysggmh7wxhh6z4rwstile3z2622jynjkfse4fezulqostrxlp.py
# Topologically Sorted Source Nodes: [conv2d_3, instance_norm_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_3
#   instance_norm_3 => add_6, rsqrt_3, var_mean_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_7, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_6, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_9 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_9', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_9(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 256*x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 256.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(in_out_ptr0 + (r2 + 256*x3), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp20, None)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/dg/cdgymnewy2wz7tasxvs5vdnunwczwmmokpyi5j5jemiw6f4wk62p.py
# Topologically Sorted Source Nodes: [instance_norm_3], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   instance_norm_3 => repeat_6
# Graph fragment:
#   %repeat_6 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_16, [4]), kwargs = {})
triton_poi_fused_repeat_10 = async_compile.triton('triton_poi_fused_repeat_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((x0 % 128)), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kk/ckksp2ak2ah66ws4sgjrjqteqjadquzi2iauob2ztcbyikaghgnn.py
# Topologically Sorted Source Nodes: [out, pad_4], Original ATen: [aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   out => relu_3
#   pad_4 => _unsafe_index_8, _unsafe_index_9
# Graph fragment:
#   %relu_3 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_7,), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_3, [None, None, %sub_16, None]), kwargs = {})
#   %_unsafe_index_9 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_8, [None, None, None, %sub_16]), kwargs = {})
triton_poi_fused_reflection_pad2d_relu_11 = async_compile.triton('triton_poi_fused_reflection_pad2d_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 165888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 18)
    x1 = ((xindex // 18) % 18)
    x2 = xindex // 324
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (255 + ((-1)*tl_math.abs((-15) + tl_math.abs((-1) + x0))) + ((-16)*tl_math.abs((-15) + tl_math.abs((-1) + x1))) + 256*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xo/cxozj6rdhwjiqseipvch37ezrjahvoqc2myytswlh5xh2cfsntre.py
# Topologically Sorted Source Nodes: [conv2d_4, out_1, y_3], Original ATen: [aten.convolution, aten.repeat, aten._native_batch_norm_legit, aten.add]
# Source node to ATen node mapping:
#   conv2d_4 => convolution_4
#   out_1 => add_8, repeat_8, rsqrt_4, var_mean_4
#   y_3 => add_10
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_9, %primals_18, %primals_19, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %repeat_8 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_20, [4]), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_8, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %relu_2), kwargs = {})
triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12 = async_compile.triton('triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr3, xnumel, rnumel):
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
    x0 = xindex
    r3 = rindex
    x1 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + ((x0 % 128)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (r3 + 256*x0), None)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_out_ptr1 + (r3 + 256*x0), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp9 = tl.full([1], 256, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 / tmp10
    tmp12 = tmp4 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = tmp3 - tmp11
    tmp18 = 256.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp24 = tmp23 * tmp0
    tmp26 = tmp24 + tmp25
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(in_out_ptr0 + (r3 + 256*x0), tmp3, None)
    tl.store(in_out_ptr1 + (r3 + 256*x0), tmp28, None)
    tl.store(out_ptr3 + (x0), tmp22, None)
    tl.store(out_ptr1 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/up/cup3mgqlzhqgd5j4eujryidxarpzscrtn6o6q6qgakuvywyvux7u.py
# Topologically Sorted Source Nodes: [conv2d_12, out_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   conv2d_12 => convolution_12
#   out_9 => add_28, rsqrt_12, var_mean_12
# Graph fragment:
#   %convolution_12 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_25, %primals_50, %primals_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
triton_per_fused__native_batch_norm_legit_convolution_13 = async_compile.triton('triton_per_fused__native_batch_norm_legit_convolution_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_convolution_13(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (r2 + 256*x3), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 256.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(in_out_ptr0 + (r2 + 256*x3), tmp2, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
    tl.store(out_ptr0 + (x3), tmp10, None)
    tl.store(out_ptr1 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/ob/coblodidnl5klszswdzmkxhzci6x52osd6mqbpxv7dasdijskzrq.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   x => iota_26
# Graph fragment:
#   %iota_26 : [num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
triton_poi_fused_arange_14 = async_compile.triton('triton_poi_fused_arange_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_14(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7f/c7f3lqh5vz72y3qembljbxarsxj6zjah5khnqsdcrrmt4fpixicf.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x => add_31, add_32, convert_element_type, convert_element_type_1, mul_26, mul_27
# Graph fragment:
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_26, 1), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_31, torch.float32), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_15 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pf/cpfjgn3jyxqshumf3pxqomp52ydcaa5fntse3igv6djthaoeiv47.py
# Topologically Sorted Source Nodes: [y_7, x, pad_13], Original ATen: [aten.add, aten._unsafe_index, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   pad_13 => _unsafe_index_27, _unsafe_index_28
#   x => _unsafe_index_26
#   y_7 => add_30
# Graph fragment:
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_25, %add_25), kwargs = {})
#   %_unsafe_index_26 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_30, [None, None, %unsqueeze_52, %convert_element_type_1]), kwargs = {})
#   %_unsafe_index_27 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_26, [None, None, %sub_11, None]), kwargs = {})
#   %_unsafe_index_28 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_27, [None, None, None, %sub_11]), kwargs = {})
triton_poi_fused__unsafe_index_add_reflection_pad2d_16 = async_compile.triton('triton_poi_fused__unsafe_index_add_reflection_pad2d_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_reflection_pad2d_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_reflection_pad2d_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 591872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 34) % 34)
    x0 = (xindex % 34)
    x4 = xindex // 1156
    x2 = ((xindex // 1156) % 128)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (31 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x1)))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (31 + ((-1)*tl_math.abs((-31) + tl_math.abs((-1) + x0)))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 16*tmp4 + 256*x4), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = 256.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp18 = tmp11 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = tl.load(in_ptr6 + (tmp8 + 16*tmp4 + 256*x4), xmask, eviction_policy='evict_last')
    tmp24 = tmp22 + tmp23
    tl.store(out_ptr0 + (x7), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6a/c6avz2ia7cglijim7pov43jzhhfusodgvmepabuvma6rw34oleny.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   x_1 => iota_30
# Graph fragment:
#   %iota_30 : [num_users=2] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
triton_poi_fused_arange_17 = async_compile.triton('triton_poi_fused_arange_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_17(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kq/ckqad6fx2x5t4bij4wtgv2gsv7nxws3rmppmevsbdnkmhk2lhn7z.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_1 => add_37, add_38, convert_element_type_4, convert_element_type_5, mul_32, mul_33
# Graph fragment:
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_30, 1), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, 0), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_37, torch.float32), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_38, 0.5), kwargs = {})
#   %convert_element_type_5 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_33, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_18 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_18(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sz/cszysne4tlrfln7aewntm5cvjyy3j4auzbvm24aq3vlv2hdli6vo.py
# Topologically Sorted Source Nodes: [y_8, x_1, pad_14], Original ATen: [aten.relu, aten._unsafe_index, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   pad_14 => _unsafe_index_30, _unsafe_index_31
#   x_1 => _unsafe_index_29
#   y_8 => relu_8
# Graph fragment:
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_27,), kwargs = {})
#   %_unsafe_index_29 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_8, [None, None, %unsqueeze_57, %convert_element_type_5]), kwargs = {})
#   %_unsafe_index_30 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_29, [None, None, %sub_6, None]), kwargs = {})
#   %_unsafe_index_31 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_30, [None, None, None, %sub_6]), kwargs = {})
triton_poi_fused__unsafe_index_reflection_pad2d_relu_19 = async_compile.triton('triton_poi_fused__unsafe_index_reflection_pad2d_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_reflection_pad2d_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_reflection_pad2d_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1115136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 66) % 66)
    x0 = (xindex % 66)
    x2 = xindex // 4356
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x1)))), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0)))), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 32*tmp4 + 1024*x2), xmask, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x5), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ug/cugcn5ogmygxmlbxjaav2oejdiyheom6rbabhqryzroqt2cwx5fo.py
# Topologically Sorted Source Nodes: [y_9, pad_15], Original ATen: [aten.relu, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   pad_15 => _unsafe_index_32, _unsafe_index_33
#   y_9 => relu_9
# Graph fragment:
#   %relu_9 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_29,), kwargs = {})
#   %_unsafe_index_32 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_9, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_33 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_32, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_relu_20 = async_compile.triton('triton_poi_fused_reflection_pad2d_relu_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 663552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 72)
    x1 = ((xindex // 72) % 72)
    x2 = xindex // 5184
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-4) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-4) + x1))) + 4096*x2), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 0, tl.int32)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/tn/ctnmdjuqjqdtgmw5hg2qq4iurjbjjybhitm2yrx53xx6blog2vow.py
# Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_15 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_33, %primals_62, %primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 9, 9), (243, 81, 9, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, ), (1, ))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (3, 32, 9, 9), (2592, 81, 9, 1))
    assert_size_stride(primals_63, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 3, 72, 72), (15552, 5184, 72, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_0.run(primals_3, buf0, 62208, grid=grid(62208), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf2 = buf1; del buf1  # reuse
        buf5 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf8 = reinterpret_tensor(buf6, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [conv2d, instance_norm], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_convolution_1.run(buf2, buf8, primals_2, buf5, 128, 4096, grid=grid(128), stream=stream0)
        del primals_2
        buf3 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(primals_4, buf3, 128, grid=grid(128), stream=stream0)
        del primals_4
        buf4 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(primals_5, buf4, 128, grid=grid(128), stream=stream0)
        del primals_5
        buf9 = empty_strided_cuda((4, 32, 66, 66), (139392, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y, pad_1], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_3.run(buf2, buf5, buf8, buf3, buf4, buf9, 557568, grid=grid(557568), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_6, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf11 = buf10; del buf10  # reuse
        buf14 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        buf15 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf17 = reinterpret_tensor(buf15, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, instance_norm_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_4.run(buf11, buf17, primals_7, buf14, 256, 1024, grid=grid(256), stream=stream0)
        del primals_7
        buf12 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_1], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_5.run(primals_8, buf12, 256, grid=grid(256), stream=stream0)
        del primals_8
        buf13 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_1], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_5.run(primals_9, buf13, 256, grid=grid(256), stream=stream0)
        del primals_9
        buf18 = empty_strided_cuda((4, 64, 34, 34), (73984, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_1, pad_2], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_6.run(buf11, buf14, buf17, buf12, buf13, buf18, 295936, grid=grid(295936), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_10, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf21 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf22 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf20 = buf19; del buf19  # reuse
        buf23 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf24 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf26 = reinterpret_tensor(buf24, (1, 512, 1, 1), (512, 1, 1, 1), 0); del buf24  # reuse
        buf27 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_2, instance_norm_2, y_2], Original ATen: [aten.convolution, aten.repeat, aten._native_batch_norm_legit, aten.relu]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_relu_repeat_7.run(buf20, buf26, primals_12, primals_13, primals_11, buf21, buf22, buf23, buf27, 512, 256, grid=grid(512), stream=stream0)
        del primals_11
        del primals_12
        del primals_13
        buf28 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_3], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_8.run(buf27, buf28, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf30 = buf29; del buf29  # reuse
        buf33 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf34 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf36 = reinterpret_tensor(buf34, (1, 512, 1, 1), (512, 1, 1, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [conv2d_3, instance_norm_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_9.run(buf30, buf36, primals_15, buf33, 512, 256, grid=grid(512), stream=stream0)
        del primals_15
        buf31 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_3], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_16, buf31, 512, grid=grid(512), stream=stream0)
        del primals_16
        buf32 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_3], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_17, buf32, 512, grid=grid(512), stream=stream0)
        del primals_17
        buf37 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out, pad_4], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_11.run(buf30, buf33, buf36, buf31, buf32, buf37, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf40 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf39 = buf38; del buf38  # reuse
        buf41 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf45 = buf27; del buf27  # reuse
        buf44 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_4, out_1, y_3], Original ATen: [aten.convolution, aten.repeat, aten._native_batch_norm_legit, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12.run(buf39, buf45, primals_20, primals_19, primals_21, buf40, buf41, buf44, 512, 256, grid=grid(512), stream=stream0)
        del primals_19
        del primals_20
        del primals_21
        buf46 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_5], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_8.run(buf45, buf46, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf48 = buf47; del buf47  # reuse
        buf51 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf52 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf54 = reinterpret_tensor(buf52, (1, 512, 1, 1), (512, 1, 1, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [conv2d_5, instance_norm_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_9.run(buf48, buf54, primals_23, buf51, 512, 256, grid=grid(512), stream=stream0)
        del primals_23
        buf49 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_5], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_24, buf49, 512, grid=grid(512), stream=stream0)
        del primals_24
        buf50 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_5], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_25, buf50, 512, grid=grid(512), stream=stream0)
        del primals_25
        buf55 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2, pad_6], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_11.run(buf48, buf51, buf54, buf49, buf50, buf55, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf58 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf57 = buf56; del buf56  # reuse
        buf59 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf63 = buf45; del buf45  # reuse
        buf62 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_6, out_3, y_4], Original ATen: [aten.convolution, aten.repeat, aten._native_batch_norm_legit, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12.run(buf57, buf63, primals_28, primals_27, primals_29, buf58, buf59, buf62, 512, 256, grid=grid(512), stream=stream0)
        del primals_27
        del primals_28
        del primals_29
        buf64 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_7], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_8.run(buf63, buf64, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf66 = buf65; del buf65  # reuse
        buf69 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf70 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf72 = reinterpret_tensor(buf70, (1, 512, 1, 1), (512, 1, 1, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [conv2d_7, instance_norm_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_9.run(buf66, buf72, primals_31, buf69, 512, 256, grid=grid(512), stream=stream0)
        del primals_31
        buf67 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_7], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_32, buf67, 512, grid=grid(512), stream=stream0)
        del primals_32
        buf68 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_7], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_33, buf68, 512, grid=grid(512), stream=stream0)
        del primals_33
        buf73 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, pad_8], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_11.run(buf66, buf69, buf72, buf67, buf68, buf73, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf76 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf75 = buf74; del buf74  # reuse
        buf77 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf81 = buf63; del buf63  # reuse
        buf80 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_8, out_5, y_5], Original ATen: [aten.convolution, aten.repeat, aten._native_batch_norm_legit, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12.run(buf75, buf81, primals_36, primals_35, primals_37, buf76, buf77, buf80, 512, 256, grid=grid(512), stream=stream0)
        del primals_35
        del primals_36
        del primals_37
        buf82 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_9], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_8.run(buf81, buf82, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf84 = buf83; del buf83  # reuse
        buf87 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf88 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf90 = reinterpret_tensor(buf88, (1, 512, 1, 1), (512, 1, 1, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [conv2d_9, instance_norm_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_9.run(buf84, buf90, primals_39, buf87, 512, 256, grid=grid(512), stream=stream0)
        del primals_39
        buf85 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_9], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_40, buf85, 512, grid=grid(512), stream=stream0)
        del primals_40
        buf86 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_9], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_41, buf86, 512, grid=grid(512), stream=stream0)
        del primals_41
        buf91 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_6, pad_10], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_11.run(buf84, buf87, buf90, buf85, buf86, buf91, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf94 = empty_strided_cuda((512, ), (1, ), torch.float32)
        buf93 = buf92; del buf92  # reuse
        buf95 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf99 = buf81; del buf81  # reuse
        buf98 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_10, out_7, y_6], Original ATen: [aten.convolution, aten.repeat, aten._native_batch_norm_legit, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_add_convolution_repeat_12.run(buf93, buf99, primals_44, primals_43, primals_45, buf94, buf95, buf98, 512, 256, grid=grid(512), stream=stream0)
        del primals_43
        del primals_44
        del primals_45
        buf100 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pad_11], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_8.run(buf99, buf100, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_11], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf102 = buf101; del buf101  # reuse
        buf105 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf106 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf108 = reinterpret_tensor(buf106, (1, 512, 1, 1), (512, 1, 1, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [conv2d_11, instance_norm_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_9.run(buf102, buf108, primals_47, buf105, 512, 256, grid=grid(512), stream=stream0)
        del primals_47
        buf103 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_11], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_48, buf103, 512, grid=grid(512), stream=stream0)
        del primals_48
        buf104 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_11], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_49, buf104, 512, grid=grid(512), stream=stream0)
        del primals_49
        buf109 = empty_strided_cuda((4, 128, 18, 18), (41472, 324, 18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8, pad_12], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_11.run(buf102, buf105, buf108, buf103, buf104, buf109, 165888, grid=grid(165888), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_50, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf111 = buf110; del buf110  # reuse
        buf113 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf114 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        buf116 = empty_strided_cuda((1, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [conv2d_12, out_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_13.run(buf111, primals_51, buf113, buf114, buf116, 512, 256, grid=grid(512), stream=stream0)
        del primals_51
        buf112 = empty_strided_cuda((512, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out_9], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_10.run(primals_52, buf112, 512, grid=grid(512), stream=stream0)
        del primals_52
        buf117 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_14.run(buf117, 32, grid=grid(32), stream=stream0)
        buf118 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_15.run(buf118, 32, grid=grid(32), stream=stream0)
        buf119 = empty_strided_cuda((4, 128, 34, 34), (147968, 1156, 34, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_7, x, pad_13], Original ATen: [aten.add, aten._unsafe_index, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_reflection_pad2d_16.run(buf118, buf111, buf113, buf114, buf112, primals_53, buf99, buf119, 591872, grid=grid(591872), stream=stream0)
        del buf114
        del buf99
        del primals_53
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf121 = buf120; del buf120  # reuse
        buf124 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        buf125 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf127 = reinterpret_tensor(buf125, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [conv2d_13, instance_norm_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_4.run(buf121, buf127, primals_55, buf124, 256, 1024, grid=grid(256), stream=stream0)
        del primals_55
        buf122 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_13], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_5.run(primals_56, buf122, 256, grid=grid(256), stream=stream0)
        del primals_56
        buf123 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_13], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_5.run(primals_57, buf123, 256, grid=grid(256), stream=stream0)
        del primals_57
        buf128 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_17.run(buf128, 64, grid=grid(64), stream=stream0)
        buf129 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_18.run(buf129, 64, grid=grid(64), stream=stream0)
        buf130 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_8, x_1, pad_14], Original ATen: [aten.relu, aten._unsafe_index, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_reflection_pad2d_relu_19.run(buf129, buf121, buf124, buf127, buf122, buf123, buf130, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 32, 64, 64), (131072, 4096, 64, 1))
        buf132 = buf131; del buf131  # reuse
        buf135 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        buf136 = empty_strided_cuda((1, 128, 1, 1), (128, 1, 128, 128), torch.float32)
        buf138 = reinterpret_tensor(buf136, (1, 128, 1, 1), (128, 1, 1, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [conv2d_14, instance_norm_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_convolution_1.run(buf132, buf138, primals_59, buf135, 128, 4096, grid=grid(128), stream=stream0)
        del primals_59
        buf133 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_14], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(primals_60, buf133, 128, grid=grid(128), stream=stream0)
        del primals_60
        buf134 = empty_strided_cuda((128, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm_14], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(primals_61, buf134, 128, grid=grid(128), stream=stream0)
        del primals_61
        buf139 = empty_strided_cuda((4, 32, 72, 72), (165888, 5184, 72, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_9, pad_15], Original ATen: [aten.relu, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_relu_20.run(buf132, buf135, buf138, buf133, buf134, buf139, 663552, grid=grid(663552), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 3, 64, 64), (12288, 4096, 64, 1))
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(buf141, primals_63, 49152, grid=grid(49152), stream=stream0)
        del primals_63
    return (buf141, primals_1, primals_6, primals_10, primals_14, primals_18, primals_22, primals_26, primals_30, primals_34, primals_38, primals_42, primals_46, primals_50, primals_54, primals_58, primals_62, buf0, buf2, buf3, buf4, buf5, buf8, buf9, buf11, buf12, buf13, buf14, buf17, buf18, buf20, buf21, buf22, buf23, buf26, buf28, buf30, buf31, buf32, buf33, buf36, buf37, buf39, buf40, reinterpret_tensor(buf44, (512, ), (1, ), 0), buf46, buf48, buf49, buf50, buf51, buf54, buf55, buf57, buf58, reinterpret_tensor(buf62, (512, ), (1, ), 0), buf64, buf66, buf67, buf68, buf69, buf72, buf73, buf75, buf76, reinterpret_tensor(buf80, (512, ), (1, ), 0), buf82, buf84, buf85, buf86, buf87, buf90, buf91, buf93, buf94, reinterpret_tensor(buf98, (512, ), (1, ), 0), buf100, buf102, buf103, buf104, buf105, buf108, buf109, buf111, buf112, reinterpret_tensor(buf116, (512, ), (1, ), 0), buf117, buf118, buf119, buf121, buf122, buf123, buf124, buf127, buf128, buf129, buf130, buf132, buf133, buf134, buf135, buf138, buf139, reinterpret_tensor(buf113, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf95, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf77, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf59, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf41, (1, 512, 1, 1), (512, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 9, 9), (243, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((3, 32, 9, 9), (2592, 81, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
