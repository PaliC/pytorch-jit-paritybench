# AOT ID: ['11_forward']
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


# kernel path: inductor_cache/73/c73bgxhspmpbjcp3lssulsjecerl42vblbm7w5yp75jalptfhui7.py
# Topologically Sorted Source Nodes: [h, input_1], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   h => convolution
#   input_1 => var_mean
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_0 = async_compile.triton('triton_red_fused_convolution_native_group_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 64)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp5, xmask)
    tl.store(out_ptr2 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ng/cngsd7ypfn6rjoncmysndkh6ypmp5xtvevjy2dc4v2pkp3iwznre.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_1 => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_per_fused_native_group_norm_1 = async_compile.triton('triton_per_fused_native_group_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 2*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 2*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 16384.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u6/cu6pdfnaqsmoregipjomjveiwuwshsh5ws2fhhjayuejdqioueyk.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_1 => add_1, mul_1
#   input_2 => mul_2, sigmoid
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
triton_poi_fused_native_group_norm_silu_2 = async_compile.triton('triton_poi_fused_native_group_norm_silu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 4096
    x1 = ((xindex // 4096) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/tw/ctwk7onfbb2otkkdko3j42tc3y2ip4cmd3bfn5rd2ufidotbq3fe.py
# Topologically Sorted Source Nodes: [input_7, input_9], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_7 => convolution_2
#   input_9 => var_mean_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_5, %primals_10, %primals_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_3 = async_compile.triton('triton_red_fused_convolution_native_group_norm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_3(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 64)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3 + tmp2
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp2, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7s/c7shqfymcqyvlthhfbc2b62bfyxf33muslrmauhiwzejldaa6siz.py
# Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_10 => mul_8, sigmoid_2
#   input_9 => add_6, mul_7
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_17), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_14), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_6,), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, %sigmoid_2), kwargs = {})
triton_poi_fused_native_group_norm_silu_4 = async_compile.triton('triton_poi_fused_native_group_norm_silu_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 4096
    x1 = ((xindex // 4096) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 // 4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/bi/cbik3i3utmn3nc6rh7s3adt2sc6brgi2nrkjk5gyt4xqy6lp6jov.py
# Topologically Sorted Source Nodes: [input_8, input_15, input_16, x], Original ATen: [aten.add, aten.convolution, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_15 => convolution_4
#   input_16 => add_9
#   input_8 => add_4
#   x => constant_pad_nd
# Graph fragment:
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_11, %primals_18, %primals_19, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_4), kwargs = {})
#   %constant_pad_nd : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_9, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_5 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2163200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 65) % 65)
    x0 = (xindex % 65)
    x4 = xindex // 4225
    x2 = ((xindex // 4225) % 128)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + 64*x1 + 4096*x4), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0 + 64*x1 + 4096*x4), tmp5 & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x0 + 64*x1 + 4096*x4), tmp5 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tl.store(out_ptr0 + (x5), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qp/cqpwa5kdhqmxz3yazipcqyze3apzbly33em3w6m325btujb2bcph.py
# Topologically Sorted Source Nodes: [x_1, input_17], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_17 => add_10, rsqrt_4, var_mean_4
#   x_1 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd, %primals_20, %primals_21, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_8, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
triton_red_fused_convolution_native_group_norm_6 = async_compile.triton('triton_red_fused_convolution_native_group_norm_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_6', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_6(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_out_ptr0 + (r5 + 4096*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 4*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 4096*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tmp7 = 4096.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lv/clva2qvditcz6ysicq46uj3uwhqvkf6u2cbo7cddohd3pekpteau.py
# Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_17 => add_11, mul_13
#   input_18 => mul_14, sigmoid_4
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %unsqueeze_29), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_26), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_11,), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, %sigmoid_4), kwargs = {})
triton_poi_fused_native_group_norm_silu_7 = async_compile.triton('triton_poi_fused_native_group_norm_silu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/b6/cb6cjppb37hlr4ayurcvj6i4gog2uvxrbcl2w2udl3y6tdie7rju.py
# Topologically Sorted Source Nodes: [input_23, input_25], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_23 => convolution_7
#   input_25 => add_15, rsqrt_6, var_mean_6
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_17, %primals_28, %primals_29, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
triton_red_fused_convolution_native_group_norm_8 = async_compile.triton('triton_red_fused_convolution_native_group_norm_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_out_ptr0 + (r5 + 4096*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 4*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 4096*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3 + tmp2
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 4096*x4), tmp2, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tmp9 = 4096.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/35/c35msxn25xtxowtahsljjdhgppq5ttcrkv76txmut3ryndqywjf3.py
# Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_25 => add_16, mul_19
#   input_26 => mul_20, sigmoid_6
# Graph fragment:
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %unsqueeze_41), kwargs = {})
#   %add_16 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_38), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_16,), kwargs = {})
#   %mul_20 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, %sigmoid_6), kwargs = {})
triton_poi_fused_native_group_norm_silu_9 = async_compile.triton('triton_poi_fused_native_group_norm_silu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 // 4), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/op/copmkanyjs244rdqi4lqkwrg7mejuvzvmzctewcpi2kyojuxozpc.py
# Topologically Sorted Source Nodes: [input_24, input_31, input_32, x_2], Original ATen: [aten.add, aten.convolution, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_24 => add_14
#   input_31 => convolution_9
#   input_32 => add_19
#   x_2 => constant_pad_nd_1
# Graph fragment:
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %convolution_7), kwargs = {})
#   %convolution_9 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_23, %primals_36, %primals_37, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %convolution_9), kwargs = {})
#   %constant_pad_nd_1 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_19, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_10 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 557568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 33) % 33)
    x0 = (xindex % 33)
    x4 = xindex // 1089
    x2 = ((xindex // 1089) % 128)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + 32*x1 + 1024*x4), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0 + 32*x1 + 1024*x4), tmp5 & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x0 + 32*x1 + 1024*x4), tmp5 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tl.store(out_ptr0 + (x5), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/da/cdarf25dp27ebcwmjsfsmz3xfvt63mfbcdgyqvqciisbqfac43fc.py
# Topologically Sorted Source Nodes: [x_3, input_33], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_33 => add_20, rsqrt_8, var_mean_8
#   x_3 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_1, %primals_38, %primals_39, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20,), kwargs = {})
triton_per_fused_convolution_native_group_norm_11 = async_compile.triton('triton_per_fused_convolution_native_group_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_11', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_11(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 256
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 1024*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 4*x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (r5 + 1024*x4), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp20, None)
    tl.store(out_ptr0 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ol/colndllsy7rky5msbtmnjynulh65hqzfgpceyovggkz4gkxtyk2g.py
# Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_33 => add_21, mul_25
#   input_34 => mul_26, sigmoid_8
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %unsqueeze_53), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_50), kwargs = {})
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_21,), kwargs = {})
#   %mul_26 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %sigmoid_8), kwargs = {})
triton_poi_fused_native_group_norm_silu_12 = async_compile.triton('triton_poi_fused_native_group_norm_silu_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/bx/cbxupx5wq7ma55hhr5fg6fdofnsdzwh3tf6g7ptayeqysozzgiyn.py
# Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_35 => convolution_11
#   input_36 => add_22, rsqrt_9, var_mean_9
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_26, %primals_42, %primals_43, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_9 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
triton_red_fused_convolution_native_group_norm_13 = async_compile.triton('triton_red_fused_convolution_native_group_norm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_13(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 256
        tmp0 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tmp7 = 2048.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckadzeig5edz7ngd556ciqiggdbxyrkoxzkpeuknsocn2l6cgrwv.py
# Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_36 => add_23, mul_28
#   input_37 => mul_29, sigmoid_9
# Graph fragment:
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %unsqueeze_59), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_28, %unsqueeze_56), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_23,), kwargs = {})
#   %mul_29 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_23, %sigmoid_9), kwargs = {})
triton_poi_fused_native_group_norm_silu_14 = async_compile.triton('triton_poi_fused_native_group_norm_silu_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/aj/cajcwhxgxltdjagjncmmamviiaem57emsq2dt2uuqlbybhwgzac3.py
# Topologically Sorted Source Nodes: [input_39, x_4, input_40, input_41], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_39 => convolution_12
#   input_40 => add_24
#   input_41 => add_25, rsqrt_10, var_mean_10
#   x_4 => convolution_13
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_29, %primals_46, %primals_47, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_13 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_10, %primals_48, %primals_49, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_13, %convolution_12), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_20, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_10 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
triton_red_fused_add_convolution_native_group_norm_15 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 256
        tmp0 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp6, rmask & xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 2048.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sd/csds4e6ftid3yql6bsk4qa4benqeay7vtodpyhsfcl2z5hppm5pb.py
# Topologically Sorted Source Nodes: [input_47, input_48, x_5], Original ATen: [aten.convolution, aten.add, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_47 => convolution_15
#   input_48 => add_29
#   x_5 => constant_pad_nd_2
# Graph fragment:
#   %convolution_15 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_35, %primals_56, %primals_57, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %convolution_15), kwargs = {})
#   %constant_pad_nd_2 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_29, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_16 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 295936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 17) % 17)
    x0 = (xindex % 17)
    x4 = xindex // 289
    x2 = ((xindex // 289) % 256)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + 16*x1 + 256*x4), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0 + 16*x1 + 256*x4), tmp5 & xmask, other=0.0)
    tmp8 = tl.load(in_ptr2 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tl.store(out_ptr0 + (x5), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ay/cay3nhaa2ykuzxondr5ykl27hrs2pfg3uoh3zutxbkbbt52n4rhg.py
# Topologically Sorted Source Nodes: [x_6, input_49], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_49 => add_30, rsqrt_12, var_mean_12
#   x_6 => convolution_16
# Graph fragment:
#   %convolution_16 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_2, %primals_58, %primals_59, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_30,), kwargs = {})
triton_per_fused_convolution_native_group_norm_17 = async_compile.triton('triton_per_fused_convolution_native_group_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_17(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 64
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 512*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(in_out_ptr0 + (r5 + 512*x4), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp20, None)
    tl.store(out_ptr0 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/hx/chxd3zmvljk4btfinlul7v22nqnahbidvsirzyaoubcckkyx7jvu.py
# Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_49 => add_31, mul_37
#   input_50 => mul_38, sigmoid_12
# Graph fragment:
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %unsqueeze_77), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, %unsqueeze_74), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_31,), kwargs = {})
#   %mul_38 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_31, %sigmoid_12), kwargs = {})
triton_poi_fused_native_group_norm_silu_18 = async_compile.triton('triton_poi_fused_native_group_norm_silu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/xi/cximmts3cedz2p7xggksot6ptv5j46u5uk26c3hdw7j6jvvembkv.py
# Topologically Sorted Source Nodes: [input_55, input_57], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_55 => convolution_18
#   input_57 => add_35, rsqrt_14, var_mean_14
# Graph fragment:
#   %convolution_18 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_41, %primals_66, %primals_67, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_14 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_28, 1e-05), kwargs = {})
#   %rsqrt_14 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
triton_per_fused_convolution_native_group_norm_19 = async_compile.triton('triton_per_fused_convolution_native_group_norm_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_19(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 64
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 512*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r5 + 512*x4), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 512.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.store(in_out_ptr0 + (r5 + 512*x4), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp22, None)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/nk/cnk3lhql2fmynriml2uksmomfqh7wxkyx4vz4if7zozakqtesmnc.py
# Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_57 => add_36, mul_43
#   input_58 => mul_44, sigmoid_14
# Graph fragment:
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_29, %unsqueeze_89), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_86), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_36,), kwargs = {})
#   %mul_44 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %sigmoid_14), kwargs = {})
triton_poi_fused_native_group_norm_silu_20 = async_compile.triton('triton_poi_fused_native_group_norm_silu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 // 8), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/fa/cfa4lva3dfxhvancoqlljnkxc4jytz3wmcyc7xpi64dmeqog4mkl.py
# Topologically Sorted Source Nodes: [input_56, input_63, input_64, x_7], Original ATen: [aten.add, aten.convolution, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   input_56 => add_34
#   input_63 => convolution_20
#   input_64 => add_39
#   x_7 => constant_pad_nd_3
# Graph fragment:
#   %add_34 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_16, %convolution_18), kwargs = {})
#   %convolution_20 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_47, %primals_74, %primals_75, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %convolution_20), kwargs = {})
#   %constant_pad_nd_3 : [num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_39, [0, 1, 0, 1], 0.0), kwargs = {})
triton_poi_fused_add_constant_pad_nd_convolution_21 = async_compile.triton('triton_poi_fused_add_constant_pad_nd_convolution_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_constant_pad_nd_convolution_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_constant_pad_nd_convolution_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 82944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 9) % 9)
    x0 = (xindex % 9)
    x4 = xindex // 81
    x2 = ((xindex // 81) % 256)
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 8, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + 8*x1 + 64*x4), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0 + 8*x1 + 64*x4), tmp5 & xmask, other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.load(in_ptr2 + (x0 + 8*x1 + 64*x4), tmp5 & xmask, other=0.0)
    tmp10 = tl.load(in_ptr3 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp5, tmp12, tmp13)
    tl.store(out_ptr0 + (x5), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cz/cczrg3qnqeuyvlmjhtsjprvndvcifvyhok7cbrsrl5gqypjyv3u7.py
# Topologically Sorted Source Nodes: [x_8, input_65], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_65 => add_40, rsqrt_16, var_mean_16
#   x_8 => convolution_21
# Graph fragment:
#   %convolution_21 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%constant_pad_nd_3, %primals_76, %primals_77, [2, 2], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_16 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_32, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_32, 1e-05), kwargs = {})
#   %rsqrt_16 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
triton_per_fused_convolution_native_group_norm_22 = async_compile.triton('triton_per_fused_convolution_native_group_norm_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_22(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 16
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 128*x4), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = 128.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(in_out_ptr0 + (r5 + 128*x4), tmp2, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp23, xmask)
    tl.store(out_ptr0 + (x4), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ms/cmstqe3qhqitduvp2kohypmgdb4x6tzlgrlspvo7v5wgqwi45xsp.py
# Topologically Sorted Source Nodes: [input_65, input_66], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_65 => add_41, mul_49
#   input_66 => mul_50, sigmoid_16
# Graph fragment:
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_33, %unsqueeze_101), kwargs = {})
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_98), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_41,), kwargs = {})
#   %mul_50 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_41, %sigmoid_16), kwargs = {})
triton_poi_fused_native_group_norm_silu_23 = async_compile.triton('triton_poi_fused_native_group_norm_silu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/td/ctdsg4rxnq7bm2puxrw2z4bewxoojxlnrku3o3jucyop6pjzfakd.py
# Topologically Sorted Source Nodes: [input_67, input_68], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_67 => convolution_22
#   input_68 => add_42, rsqrt_17, var_mean_17
# Graph fragment:
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_50, %primals_80, %primals_81, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_17 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_34, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_34, 1e-05), kwargs = {})
#   %rsqrt_17 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_42,), kwargs = {})
triton_per_fused_convolution_native_group_norm_24 = async_compile.triton('triton_per_fused_convolution_native_group_norm_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_24(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 16
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 256*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (r5 + 256*x4), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp20, None)
    tl.store(out_ptr0 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/kg/ckgps766yqclveda2ngrad5fadaovk5uch63okoghhap7ksvkj7q.py
# Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_68 => add_43, mul_52
#   input_69 => mul_53, sigmoid_17
# Graph fragment:
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_35, %unsqueeze_107), kwargs = {})
#   %add_43 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %unsqueeze_104), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_43,), kwargs = {})
#   %mul_53 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_43, %sigmoid_17), kwargs = {})
triton_poi_fused_native_group_norm_silu_25 = async_compile.triton('triton_poi_fused_native_group_norm_silu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 16
    x1 = ((xindex // 16) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 16), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 16), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ch/cchgcmp3rtddbc3li4e3lpxurovan2lz5tu2fhfzae6zmotj4ya5.py
# Topologically Sorted Source Nodes: [input_71, x_9, input_72, group_norm_18], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_18 => add_45, rsqrt_18, var_mean_18
#   input_71 => convolution_23
#   input_72 => add_44
#   x_9 => convolution_24
# Graph fragment:
#   %convolution_23 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_53, %primals_84, %primals_85, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_21, %primals_86, %primals_87, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_24, %convolution_23), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_36, 1e-05), kwargs = {})
#   %rsqrt_18 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_45,), kwargs = {})
triton_per_fused_add_convolution_native_group_norm_26 = async_compile.triton('triton_per_fused_add_convolution_native_group_norm_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_native_group_norm_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_native_group_norm_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 16
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 256*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r5 + 256*x4), None)
    tmp4 = tl.load(in_ptr2 + (r3 + 16*x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tl.full([1], 256, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp7 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 256.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tl.store(in_out_ptr0 + (r5 + 256*x4), tmp6, None)
    tl.store(out_ptr2 + (x4), tmp24, None)
    tl.store(out_ptr0 + (x4), tmp14, None)
    tl.store(out_ptr1 + (x4), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/zu/czuwiecqcgwoxntkrxikx2iquhuyvzp66ikodxf5qw33jd3y6a4c.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_9
# Graph fragment:
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_27 = async_compile.triton('triton_poi_fused_clone_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex
    x1 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 // 16), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x3 // 16), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + 2048*y0), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ct/cctcdg2ertgdsu7tpr4c2jxiqzxt5fc4v3mrot5dxgrgxeuess2y.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_10
# Graph fragment:
#   %clone_10 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze_38,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_28 = async_compile.triton('triton_poi_fused_clone_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 64)
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 1536*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/xy/cxy5co6lfnrhqidkmpzpotmpzybv6gjxzfb6bcej5w2r6634i2yv.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_11
# Graph fragment:
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_29 = async_compile.triton('triton_poi_fused_clone_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_29(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 512)
    x1 = ((xindex // 512) % 4)
    x2 = xindex // 2048
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 8192*x1), None)
    tl.store(out_ptr0 + (x3), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/mb/cmbqsxtocpa5jk2mgqfw3y2rmgqpsdgbesqoiacswh7l625pkfv5.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   out_2 => add_48
# Graph fragment:
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %permute_10), kwargs = {})
triton_poi_fused_add_30 = async_compile.triton('triton_poi_fused_add_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3 + 2048*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(out_ptr0 + (x2 + 16*y3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/67/c673tnvxiutuprgwlwvoc3qn5vtwqe25rpvyjvva756ecrpenuhe.py
# Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_73 => add_49, rsqrt_19, var_mean_19
# Graph fragment:
#   %var_mean_19 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_51, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_42, 1e-05), kwargs = {})
#   %rsqrt_19 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_49,), kwargs = {})
triton_per_fused_native_group_norm_31 = async_compile.triton('triton_per_fused_native_group_norm_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_31(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
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
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 256, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 256.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/72/c72dl646vhvhelbi45r545qsyweljxno23r5fb437fkvpmjrpsbl.py
# Topologically Sorted Source Nodes: [input_79, input_80, group_norm_21], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_21 => add_54, rsqrt_21, var_mean_21
#   input_79 => convolution_26
#   input_80 => add_53
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_61, %primals_100, %primals_101, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_53 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_48, %convolution_26), kwargs = {})
#   %var_mean_21 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_55, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-05), kwargs = {})
#   %rsqrt_21 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_54,), kwargs = {})
triton_per_fused_add_convolution_native_group_norm_32 = async_compile.triton('triton_per_fused_add_convolution_native_group_norm_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_native_group_norm_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_native_group_norm_32(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 16
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (r5 + 256*x4), None)
    tmp1 = tl.load(in_out_ptr0 + (r5 + 256*x4), None)
    tmp2 = tl.load(in_ptr1 + (r3 + 16*x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
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
    tl.store(in_out_ptr0 + (r5 + 256*x4), tmp4, None)
    tl.store(out_ptr2 + (x4), tmp22, None)
    tl.store(out_ptr0 + (x4), tmp12, None)
    tl.store(out_ptr1 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cikv2dry4gkwlcxf7ro3xm5cbotebtapt5jeice2rzrwl4t2alzl.py
# Topologically Sorted Source Nodes: [dist], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   dist => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_136, %full_default_1, %sum_2], -1), kwargs = {})
triton_poi_fused__euclidean_dist_33 = async_compile.triton('triton_poi_fused__euclidean_dist_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 5, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = 1.0
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp0 >= tmp7
    tmp14 = tl.full([1], 6, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr0 + (4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 * tmp16
    tmp18 = tl.load(in_ptr0 + (1 + 4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp18 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = tl.load(in_ptr0 + (2 + 4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 * tmp21
    tmp23 = tmp20 + tmp22
    tmp24 = tl.load(in_ptr0 + (3 + 4*x1), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp13, tmp26, tmp27)
    tmp29 = tl.where(tmp9, tmp12, tmp28)
    tmp30 = tl.where(tmp4, tmp5, tmp29)
    tl.store(out_ptr0 + (x2), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ij/cijsouv4joyvzatlod2xehdrx3ixodgw74yq7tev6yn24ei6kobw.py
# Topologically Sorted Source Nodes: [input_95, input_96, input_97], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_95 => convolution_30
#   input_96 => add_71
#   input_97 => add_72, rsqrt_27, var_mean_27
# Graph fragment:
#   %convolution_30 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_77, %primals_128, %primals_129, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_71 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_66, %convolution_30), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_93, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_66, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_72,), kwargs = {})
triton_per_fused_add_convolution_native_group_norm_34 = async_compile.triton('triton_per_fused_add_convolution_native_group_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_native_group_norm_34', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_native_group_norm_34(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 16
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (r5 + 256*x4), None)
    tmp1 = tl.load(in_out_ptr0 + (r5 + 256*x4), None)
    tmp2 = tl.load(in_ptr1 + (r3 + 16*x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
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
    tl.store(in_out_ptr0 + (r5 + 256*x4), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp22, None)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/eo/ceo2w4polvfsddbfqq2p46fj6cfmnqqdaflw5cr64k6flcvkuotk.py
# Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_99 => convolution_31
# Graph fragment:
#   %convolution_31 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_80, %primals_132, %primals_133, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_35 = async_compile.triton('triton_poi_fused_convolution_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/nl/cnlb5cisoydgvzllniwjf6s72eltowwdggdnd77wzmt36gepz4jk.py
# Topologically Sorted Source Nodes: [dist], Original ATen: [aten._euclidean_dist]
# Source node to ATen node mapping:
#   dist => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_81, %sum_1, %full_default], -1), kwargs = {})
triton_poi_fused__euclidean_dist_36 = async_compile.triton('triton_poi_fused__euclidean_dist_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 6)
    x1 = xindex // 6
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (16*(x0) + 64*(x1 // 16) + ((x1 % 16))), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = -2.0
    tmp7 = tmp5 * tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 5, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr0 + (64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp14 * tmp14
    tmp16 = tl.load(in_ptr0 + (16 + 64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = tl.load(in_ptr0 + (32 + 64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tl.load(in_ptr0 + (48 + 64*(x1 // 16) + ((x1 % 16))), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp13, tmp24, tmp25)
    tmp27 = tmp0 >= tmp11
    tmp28 = tl.full([1], 6, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = 1.0
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp27, tmp30, tmp31)
    tmp33 = tl.where(tmp13, tmp26, tmp32)
    tmp34 = tl.where(tmp4, tmp9, tmp33)
    tl.store(out_ptr0 + (x2), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z2/cz26sika7zefp5lojwx37h274payzyys5oxpsq5mb6z2wlbiulxr.py
# Topologically Sorted Source Nodes: [dist, argmin], Original ATen: [aten._euclidean_dist, aten.argmin]
# Source node to ATen node mapping:
#   argmin => argmin
#   dist => clamp_min, sqrt
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mm_3, 0), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%clamp_min,), kwargs = {})
#   %argmin : [num_users=1] = call_function[target=torch.ops.aten.argmin.default](args = (%sqrt, 1), kwargs = {})
triton_poi_fused__euclidean_dist_argmin_37 = async_compile.triton('triton_poi_fused__euclidean_dist_argmin_37', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__euclidean_dist_argmin_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__euclidean_dist_argmin_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp3 = libdevice.sqrt(tmp2)
    tmp5 = triton_helpers.maximum(tmp4, tmp1)
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp3 == tmp6
    tmp9 = tmp3 != tmp3
    tmp10 = tmp6 != tmp6
    tmp11 = tmp9 > tmp10
    tmp12 = tmp7 | tmp11
    tmp13 = tmp9 & tmp10
    tmp14 = tmp8 | tmp13
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tl.full([1], 1, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp14 & tmp17
    tmp19 = tmp12 | tmp18
    tmp20 = tl.where(tmp19, tmp3, tmp6)
    tmp21 = tl.where(tmp19, tmp15, tmp16)
    tmp23 = triton_helpers.maximum(tmp22, tmp1)
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = tmp20 < tmp24
    tmp26 = tmp20 == tmp24
    tmp27 = tmp20 != tmp20
    tmp28 = tmp24 != tmp24
    tmp29 = tmp27 > tmp28
    tmp30 = tmp25 | tmp29
    tmp31 = tmp27 & tmp28
    tmp32 = tmp26 | tmp31
    tmp33 = tl.full([1], 2, tl.int64)
    tmp34 = tmp21 < tmp33
    tmp35 = tmp32 & tmp34
    tmp36 = tmp30 | tmp35
    tmp37 = tl.where(tmp36, tmp20, tmp24)
    tmp38 = tl.where(tmp36, tmp21, tmp33)
    tmp40 = triton_helpers.maximum(tmp39, tmp1)
    tmp41 = libdevice.sqrt(tmp40)
    tmp42 = tmp37 < tmp41
    tmp43 = tmp37 == tmp41
    tmp44 = tmp37 != tmp37
    tmp45 = tmp41 != tmp41
    tmp46 = tmp44 > tmp45
    tmp47 = tmp42 | tmp46
    tmp48 = tmp44 & tmp45
    tmp49 = tmp43 | tmp48
    tmp50 = tl.full([1], 3, tl.int64)
    tmp51 = tmp38 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tmp47 | tmp52
    tmp54 = tl.where(tmp53, tmp37, tmp41)
    tmp55 = tl.where(tmp53, tmp38, tmp50)
    tl.store(out_ptr0 + (x0), tmp55, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a6/ca6ld2ugwimh236m6pyhmlltcdt5leqjexaolvbet6qegqsjuiki.py
# Topologically Sorted Source Nodes: [embeddings], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embeddings => embedding
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_136, %view_99), kwargs = {})
triton_poi_fused_embedding_38 = async_compile.triton('triton_poi_fused_embedding_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_38(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (x0 + 4*tmp4), xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qg/cqg6xwrlzbotkwb3rnafsqssditaqqebvljhvpdtcwacvalcwfx7.py
# Topologically Sorted Source Nodes: [sub, x_q_1], Original ATen: [aten.sub, aten.add]
# Source node to ATen node mapping:
#   sub => sub_30
#   x_q_1 => add_75
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_35, %convolution_32), kwargs = {})
#   %add_75 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_32, %sub_30), kwargs = {})
triton_poi_fused_add_sub_39 = async_compile.triton('triton_poi_fused_add_sub_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sub_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sub_39(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (y0 + 4*x2 + 64*y1), xmask & ymask)
    tmp2 = tmp1 - tmp0
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x2 + 16*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jj/cjjd7wdaxueslmw3ldo7ghwyfp5ltjjjrxxszrtl3qcrtruxtfqj.py
# Topologically Sorted Source Nodes: [input_107, group_norm_30], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   group_norm_30 => add_81, rsqrt_30, var_mean_30
#   input_107 => convolution_36
# Graph fragment:
#   %convolution_36 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_88, %primals_147, %primals_148, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_30 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_104, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_81 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-05), kwargs = {})
#   %rsqrt_30 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_81,), kwargs = {})
triton_per_fused_convolution_native_group_norm_40 = async_compile.triton('triton_per_fused_convolution_native_group_norm_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_40(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 16
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 256*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r5 + 256*x4), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3 + tmp2
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
    tl.store(in_out_ptr0 + (r5 + 256*x4), tmp2, None)
    tl.store(out_ptr2 + (x4), tmp22, None)
    tl.store(out_ptr0 + (x4), tmp12, None)
    tl.store(out_ptr1 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/md/cmdkxa557rreg7js7t62j4e622sxzgh6exkvh7az5glbbx3sgxa7.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward_3 => clone_23
# Graph fragment:
#   %clone_23 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_37,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_41 = async_compile.triton('triton_poi_fused_clone_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2 + 4*(((x2 % 4)) // 4) + 16*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 4*(((x2 % 4)) // 4) + 16*y3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3 // 16), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y3 // 16), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 256.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (y3 + 2048*x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qu/cqu6pfeqhesuehadha3fbvu5ike2tefs5qdw24uibtx4gnv7njy7.py
# Topologically Sorted Source Nodes: [input_108, out_11], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   input_108 => add_80
#   out_11 => add_84
# Graph fragment:
#   %add_80 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_34, %convolution_36), kwargs = {})
#   %add_84 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_80, %permute_46), kwargs = {})
triton_poi_fused_add_42 = async_compile.triton('triton_poi_fused_add_42', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 16*y3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y3 + 2048*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x2 + 16*y3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rd/crdvhhin23a2rtnftcd4ppnuijcxb32p2gfacysihxrkcvwpla2n.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_10 => add_117, add_118, convert_element_type, convert_element_type_1, iota, mul_121, mul_122
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_121, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_117, torch.float32), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_118, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_122, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_43 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_43(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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


# kernel path: inductor_cache/fp/cfpq4ckdpgtvvbrobntgr5aeibvwfunf3jngxqgs2dxwvxegsazx.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_12 => add_136, add_137, convert_element_type_4, convert_element_type_5, iota_2, mul_143, mul_144
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_2, 1), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_143, 0), kwargs = {})
#   %convert_element_type_4 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_136, torch.float32), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_137, 0.5), kwargs = {})
#   %convert_element_type_5 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_144, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_44 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_44(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: inductor_cache/7l/c7l4ylvwk72vtg3knekv6f3ye4dwyltokneprz3jaxve4psudlym.py
# Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_13 => add_155, add_156, convert_element_type_8, convert_element_type_9, iota_4, mul_165, mul_166
# Graph fragment:
#   %iota_4 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_4, 1), kwargs = {})
#   %add_155 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_165, 0), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_155, torch.float32), kwargs = {})
#   %add_156 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_8, 0.0), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, 0.5), kwargs = {})
#   %convert_element_type_9 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_166, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_45 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_45(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/3b/c3b2cmfjiplyppbtql6sezx7dum3uoc2cg22xtpgw2ie36jarmcm.py
# Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_15 => add_174, add_175, convert_element_type_12, convert_element_type_13, iota_6, mul_187, mul_188
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_174 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_187, 0), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_174, torch.float32), kwargs = {})
#   %add_175 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_12, 0.0), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_175, 0.5), kwargs = {})
#   %convert_element_type_13 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_188, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_46 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_46(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ft/cftlmlazzv4f4jjxtgfh3fgvua4pupg2rnp2lx4s5whvozgslogf.py
# Topologically Sorted Source Nodes: [out_20, x_10], Original ATen: [aten.add, aten._unsafe_index]
# Source node to ATen node mapping:
#   out_20 => add_116
#   x_10 => _unsafe_index
# Graph fragment:
#   %add_116 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_112, %permute_79), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_116, [None, None, %unsqueeze_259, %convert_element_type_1]), kwargs = {})
triton_poi_fused__unsafe_index_add_47 = async_compile.triton('triton_poi_fused__unsafe_index_add_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x5 = xindex // 64
    x2 = ((xindex // 64) % 512)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 4*tmp4 + 16*x5), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x5 + 2048*tmp8 + 8192*tmp4), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tl.store(out_ptr0 + (x6), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/s2/cs2ro7l6di2n3vnemz4kvhmj7asboj4tyilka2vuqwnkx7bxnpbc.py
# Topologically Sorted Source Nodes: [input_141, input_142], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_141 => convolution_45
#   input_142 => add_121, rsqrt_42, var_mean_42
# Graph fragment:
#   %convolution_45 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index, %primals_205, %primals_206, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_42 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_180, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_112, 1e-05), kwargs = {})
#   %rsqrt_42 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_121,), kwargs = {})
triton_per_fused_convolution_native_group_norm_48 = async_compile.triton('triton_per_fused_convolution_native_group_norm_48', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_native_group_norm_48', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_convolution_native_group_norm_48(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 64
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 1024*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (r5 + 1024*x4), tmp2, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp20, None)
    tl.store(out_ptr0 + (x4), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/dw/cdwpttjztbx7yzsfnpm2w6hbhgpbw3imytzh26trnjxay2pehwgk.py
# Topologically Sorted Source Nodes: [input_142, input_143], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_142 => add_122, mul_126
#   input_143 => mul_127, sigmoid_35
# Graph fragment:
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_181, %unsqueeze_265), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_126, %unsqueeze_262), kwargs = {})
#   %sigmoid_35 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_127 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_35), kwargs = {})
triton_poi_fused_native_group_norm_silu_49 = async_compile.triton('triton_poi_fused_native_group_norm_silu_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 16), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 16), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/fk/cfkyjefhfw3licpk7agvdvm3wnh3yazi3oupqaowmiud4fm2mvzz.py
# Topologically Sorted Source Nodes: [input_148, x_11, input_149, input_150], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_148 => convolution_47
#   input_149 => add_125
#   input_150 => add_126, rsqrt_44, var_mean_44
#   x_11 => convolution_48
# Graph fragment:
#   %convolution_47 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_130, %primals_213, %primals_214, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_48 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_45, %primals_215, %primals_216, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_125 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_48, %convolution_47), kwargs = {})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_184, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_116, 1e-05), kwargs = {})
#   %rsqrt_44 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_126,), kwargs = {})
triton_per_fused_add_convolution_native_group_norm_50 = async_compile.triton('triton_per_fused_add_convolution_native_group_norm_50', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_native_group_norm_50', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_native_group_norm_50(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 64
    x0 = (xindex % 32)
    tmp0 = tl.load(in_out_ptr0 + (r5 + 512*x4), None)
    tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (r5 + 512*x4), None)
    tmp4 = tl.load(in_ptr2 + (r3 + 8*x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tl.full([1], 512, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp7 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = 512.0
    tmp21 = tmp19 / tmp20
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tl.store(in_out_ptr0 + (r5 + 512*x4), tmp6, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp24, None)
    tl.store(out_ptr0 + (x4), tmp14, None)
''', device_str='cuda')


# kernel path: inductor_cache/ti/ctie6lor6nsypcsa6afvi7tftaq5k64bv5dyo3hhbl3cszmu726i.py
# Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_156 => convolution_50
#   input_157 => add_130
#   input_158 => add_131, rsqrt_46, var_mean_46
# Graph fragment:
#   %convolution_50 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_136, %primals_223, %primals_224, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_130 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_125, %convolution_50), kwargs = {})
#   %var_mean_46 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_188, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_120, 1e-05), kwargs = {})
#   %rsqrt_46 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_131,), kwargs = {})
triton_per_fused_add_convolution_native_group_norm_51 = async_compile.triton('triton_per_fused_add_convolution_native_group_norm_51', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_native_group_norm_51', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_convolution_native_group_norm_51(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r5 = rindex
    x4 = xindex
    r3 = rindex // 64
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (r5 + 512*x4), None)
    tmp1 = tl.load(in_out_ptr0 + (r5 + 512*x4), None)
    tmp2 = tl.load(in_ptr1 + (r3 + 8*x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tl.full([1], 512, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp5 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp18 = 512.0
    tmp19 = tmp17 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tl.store(in_out_ptr0 + (r5 + 512*x4), tmp4, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp22, None)
    tl.store(out_ptr0 + (x4), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/f2/cf2w2g4kcxubc4c2h2plrs6r7joihnita5f2y6eukgzh2xqvuzx5.py
# Topologically Sorted Source Nodes: [input_164, input_165, x_12], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_164 => convolution_52
#   input_165 => add_135
#   x_12 => _unsafe_index_1
# Graph fragment:
#   %convolution_52 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_142, %primals_231, %primals_232, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_130, %convolution_52), kwargs = {})
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_135, [None, None, %unsqueeze_296, %convert_element_type_5]), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_52 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex // 256
    x2 = ((xindex // 256) % 256)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 8*tmp4 + 64*x5), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 8*tmp4 + 64*x5), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tl.store(out_ptr0 + (x6), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crb57fx6bm4scrby7uqffpp7zhzi4fzn4tnqxyufsfzxpnobxfde.py
# Topologically Sorted Source Nodes: [input_173, input_175], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_173 => convolution_55
#   input_175 => add_145, rsqrt_50, var_mean_50
# Graph fragment:
#   %convolution_55 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_152, %primals_241, %primals_242, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_50 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_196, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_145 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_128, 1e-05), kwargs = {})
#   %rsqrt_50 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_145,), kwargs = {})
triton_red_fused_convolution_native_group_norm_53 = async_compile.triton('triton_red_fused_convolution_native_group_norm_53', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_53', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_53(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 256
        tmp0 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3 + tmp2
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp2, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tmp9 = 2048.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuuhv354nprrwv5qesb3vp7nrf3s2niekwqnw54xomgrd3gauuaj.py
# Topologically Sorted Source Nodes: [input_175, input_176], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_175 => add_146, mul_154
#   input_176 => mul_155, sigmoid_43
# Graph fragment:
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_197, %unsqueeze_314), kwargs = {})
#   %add_146 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_154, %unsqueeze_311), kwargs = {})
#   %sigmoid_43 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_146,), kwargs = {})
#   %mul_155 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_146, %sigmoid_43), kwargs = {})
triton_poi_fused_native_group_norm_silu_54 = async_compile.triton('triton_poi_fused_native_group_norm_silu_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 256
    x1 = ((xindex // 256) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x4 // 8), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sigmoid(tmp10)
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/xi/cxirgymlacytgi4dnezqkpbfaj2adsefzhwarwsxdxz5ggnesjdh.py
# Topologically Sorted Source Nodes: [input_174, input_181, input_182, input_183], Original ATen: [aten.add, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_174 => add_144
#   input_181 => convolution_57
#   input_182 => add_149
#   input_183 => add_150, rsqrt_52, var_mean_52
# Graph fragment:
#   %add_144 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_53, %convolution_55), kwargs = {})
#   %convolution_57 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_158, %primals_249, %primals_250, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_149 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_144, %convolution_57), kwargs = {})
#   %var_mean_52 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_200, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_150 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_132, 1e-05), kwargs = {})
#   %rsqrt_52 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_150,), kwargs = {})
triton_red_fused_add_convolution_native_group_norm_55 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_55', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_55', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_55(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 256
        tmp0 = tl.load(in_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r5 + 2048*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 2048*x4), tmp6, rmask & xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 2048.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7x/c7xk73w2ug6qqawi63q7ldnlstkbg5dz4uv2no3ry7n2cdrrzt7x.py
# Topologically Sorted Source Nodes: [input_189, input_190, x_13], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_189 => convolution_59
#   input_190 => add_154
#   x_13 => _unsafe_index_2
# Graph fragment:
#   %convolution_59 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_164, %primals_257, %primals_258, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_154 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_149, %convolution_59), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_154, [None, None, %unsqueeze_333, %convert_element_type_9]), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_56 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_56', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x5 = xindex // 1024
    x2 = ((xindex // 1024) % 256)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 16, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 16*tmp4 + 256*x5), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 16*tmp4 + 256*x5), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tl.store(out_ptr0 + (x6), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/d2/cd26ptbbhfekycu5iadznbkwrwo5uzblpb5tbkoyhl6yk6auykdx.py
# Topologically Sorted Source Nodes: [input_191, input_192], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_191 => convolution_60
#   input_192 => add_159, rsqrt_54, var_mean_54
# Graph fragment:
#   %convolution_60 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_2, %primals_259, %primals_260, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %var_mean_54 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_204, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_136, 1e-05), kwargs = {})
#   %rsqrt_54 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_159,), kwargs = {})
triton_red_fused_convolution_native_group_norm_57 = async_compile.triton('triton_red_fused_convolution_native_group_norm_57', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_57', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_57(in_out_ptr0, in_out_ptr1, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tmp7 = 8192.0
    tmp8 = tmp5 / tmp7
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a7/ca7sm65hrfyvfr447dh3tlovvnm7b7vd2dirgdef5geeeoliqh7p.py
# Topologically Sorted Source Nodes: [input_192, input_193], Original ATen: [aten.native_group_norm, aten.silu]
# Source node to ATen node mapping:
#   input_192 => add_160, mul_170
#   input_193 => mul_171, sigmoid_47
# Graph fragment:
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_205, %unsqueeze_339), kwargs = {})
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_170, %unsqueeze_336), kwargs = {})
#   %sigmoid_47 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_160,), kwargs = {})
#   %mul_171 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_160, %sigmoid_47), kwargs = {})
triton_poi_fused_native_group_norm_silu_58 = async_compile.triton('triton_poi_fused_native_group_norm_silu_58', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_silu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_silu_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ai/caiun5hiwkrluunf3gof6blagazaklxt6425qyg3zkjdey5ltjgf.py
# Topologically Sorted Source Nodes: [input_198, x_14, input_199, input_200], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_198 => convolution_62
#   input_199 => add_163
#   input_200 => add_164, rsqrt_56, var_mean_56
#   x_14 => convolution_63
# Graph fragment:
#   %convolution_62 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_174, %primals_267, %primals_268, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_63 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_60, %primals_269, %primals_270, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_163 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_63, %convolution_62), kwargs = {})
#   %var_mean_56 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_208, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_140, 1e-05), kwargs = {})
#   %rsqrt_56 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_164,), kwargs = {})
triton_red_fused_add_convolution_native_group_norm_59 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_59', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_59', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_59(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_out_ptr0 + (r5 + 4096*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 4*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 4096*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3 + 4*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 4096*x4), tmp6, rmask & xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tmp11 = 4096.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yr/cyr4hpfzzexlcnx6z532zwrnglqpfy3vnef2x554n52m776j26c4.py
# Topologically Sorted Source Nodes: [input_206, input_207, input_208], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_206 => convolution_65
#   input_207 => add_168
#   input_208 => add_169, rsqrt_58, var_mean_58
# Graph fragment:
#   %convolution_65 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_180, %primals_277, %primals_278, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_168 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %convolution_65), kwargs = {})
#   %var_mean_58 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_212, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_144, 1e-05), kwargs = {})
#   %rsqrt_58 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_169,), kwargs = {})
triton_red_fused_add_convolution_native_group_norm_60 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_60', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_60', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_60(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 32)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 1024
        tmp0 = tl.load(in_ptr0 + (r5 + 4096*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r5 + 4096*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + 4*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 4096*x4), tmp4, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tmp9 = 4096.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bu/cbuft2o25nyn5ah66zbqzxkt6p6utdsxswdtmjmia6uwmgqp7asi.py
# Topologically Sorted Source Nodes: [input_214, input_215, x_15], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_214 => convolution_67
#   input_215 => add_173
#   x_15 => _unsafe_index_3
# Graph fragment:
#   %convolution_67 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_186, %primals_285, %primals_286, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_173 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_168, %convolution_67), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_173, [None, None, %unsqueeze_370, %convert_element_type_13]), kwargs = {})
triton_poi_fused__unsafe_index_add_convolution_61 = async_compile.triton('triton_poi_fused__unsafe_index_add_convolution_61', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_convolution_61', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_convolution_61(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex // 4096
    x2 = ((xindex // 4096) % 128)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 32*tmp4 + 1024*x5), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (tmp8 + 32*tmp4 + 1024*x5), None, eviction_policy='evict_last')
    tmp12 = tmp10 + tmp11
    tmp13 = tmp9 + tmp12
    tl.store(out_ptr0 + (x6), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/4p/c4prxc3mlrtluepzy46rwzvacljp7t4o4jswbqyrtmlrsza2reyv.py
# Topologically Sorted Source Nodes: [input_224, input_231, input_232, input_233], Original ATen: [aten.add, aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_224 => add_182
#   input_231 => convolution_72
#   input_232 => add_187
#   input_233 => var_mean_64
# Graph fragment:
#   %add_182 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_68, %convolution_70), kwargs = {})
#   %convolution_72 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_202, %primals_303, %primals_304, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_187 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_182, %convolution_72), kwargs = {})
#   %var_mean_64 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_224, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_convolution_native_group_norm_62 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_62', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 64)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp6, rmask & xmask)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
    tl.store(out_ptr2 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck3pestybr5hund6gj36zztnwb2vzmar3d7zyn5eiielzo2k377r.py
# Topologically Sorted Source Nodes: [input_239, input_240, input_241], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_239 => convolution_74
#   input_240 => add_192
#   input_241 => var_mean_66
# Graph fragment:
#   %convolution_74 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_208, %primals_311, %primals_312, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_192 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_187, %convolution_74), kwargs = {})
#   %var_mean_66 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_228, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_convolution_native_group_norm_63 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_63', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_63', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_63(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 64)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp4, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ig/cigmmao62umhrj4e7x3c3v5w2bs4s6gr6xb6mxfbyyh4mjcsoznu.py
# Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_243 => convolution_75
# Graph fragment:
#   %convolution_75 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_211, %primals_315, %primals_316, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_64 = async_compile.triton('triton_poi_fused_convolution_64', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_64(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/a3/ca3zrlysafc4m2ra7v47fgpta33ntdw7owex3qn7ecaavmpchmfx.py
# Topologically Sorted Source Nodes: [input_244, input_245], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_244 => convolution_76
#   input_245 => gt, mul_212, where
# Graph fragment:
#   %convolution_76 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%convolution_75, %primals_317, %primals_318, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_76, 0), kwargs = {})
#   %mul_212 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_76, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_76, %mul_212), kwargs = {})
triton_poi_fused_convolution_leaky_relu_65 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_65', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_65(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.2
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chwuj7ontmctfrbdhffh4ugp24fe5y5aspwr2spg6il5i4qdkd3x.py
# Topologically Sorted Source Nodes: [input_247, input_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_247 => add_196, mul_214, mul_215, sub_70
#   input_248 => gt_1, mul_216, where_1
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_414), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_416), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_214, %unsqueeze_418), kwargs = {})
#   %add_196 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_215, %unsqueeze_420), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_196, 0), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_196, 0.2), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_196, %mul_216), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_66 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_66', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_66', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_66(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.2
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/av/cavf4ccj2uw76q2x7rvhrbq43ifq3mppdlnedw6ft4qpmutblvm6.py
# Topologically Sorted Source Nodes: [input_250, input_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_250 => add_198, mul_218, mul_219, sub_71
#   input_251 => gt_2, mul_220, where_2
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_422), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_424), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %unsqueeze_426), kwargs = {})
#   %add_198 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_219, %unsqueeze_428), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_198, 0), kwargs = {})
#   %mul_220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_198, 0.2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_198, %mul_220), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_67', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_67(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.2
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/ps/cpsjtl5bbu4t3q4lzmvqguzd72emzdzuh2z7rh3i46bupkhbjxuq.py
# Topologically Sorted Source Nodes: [input_253, input_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_253 => add_200, mul_222, mul_223, sub_72
#   input_254 => gt_3, mul_224, where_3
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_79, %unsqueeze_430), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_432), kwargs = {})
#   %mul_223 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_222, %unsqueeze_434), kwargs = {})
#   %add_200 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_223, %unsqueeze_436), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_200, 0), kwargs = {})
#   %mul_224 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_200, 0.2), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_200, %mul_224), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_68 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_68', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_68(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 49) % 512)
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
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp18 = 0.2
    tmp19 = tmp15 * tmp18
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xz/cxzrvi2tivddoffcfp4f2yfa3bagrinxj4omhr2bidj6ldo7mmiy.py
# Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   out_21 => convolution_80
# Graph fragment:
#   %convolution_80 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%where_3, %primals_334, %primals_335, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_69 = async_compile.triton('triton_poi_fused_convolution_69', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_69', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_69(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vs/cvsrmml3atoq3ltt6obb4eb5l4ujh7mxdz7ekax2spszha6ngis4.py
# Topologically Sorted Source Nodes: [mse_loss, mul, loss], Original ATen: [aten.mse_loss, aten.mul, aten.add]
# Source node to ATen node mapping:
#   loss => add_74
#   mse_loss => mean, pow_3, sub_28
#   mul => mul_82
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %permute_35), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_28, 2), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 0.2), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, %mul_82), kwargs = {})
triton_per_fused_add_mse_loss_mul_70 = async_compile.triton('triton_per_fused_add_mse_loss_mul_70', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mse_loss_mul_70', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mse_loss_mul_70(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r3 = rindex
    r0 = (rindex % 16)
    r1 = ((rindex // 16) % 4)
    r2 = rindex // 64
    tmp0 = tl.load(in_ptr0 + (r3), None)
    tmp1 = tl.load(in_ptr1 + (r1 + 4*r0 + 64*r2), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp11 = tmp8 + tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp11, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335 = args
    args.clear()
    assert_size_stride(primals_1, (128, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (1536, ), (1, ))
    assert_size_stride(primals_91, (1536, 512), (512, 1))
    assert_size_stride(primals_92, (512, 512), (512, 1))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (1536, ), (1, ))
    assert_size_stride(primals_105, (1536, 512), (512, 1))
    assert_size_stride(primals_106, (512, 512), (512, 1))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_112, (512, ), (1, ))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (1536, ), (1, ))
    assert_size_stride(primals_119, (1536, 512), (512, 1))
    assert_size_stride(primals_120, (512, 512), (512, 1))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (512, ), (1, ))
    assert_size_stride(primals_132, (4, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_133, (4, ), (1, ))
    assert_size_stride(primals_134, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_135, (4, ), (1, ))
    assert_size_stride(primals_136, (4, 4), (4, 1))
    assert_size_stride(primals_137, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_138, (4, ), (1, ))
    assert_size_stride(primals_139, (512, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_141, (512, ), (1, ))
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_143, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, ), (1, ))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (1536, ), (1, ))
    assert_size_stride(primals_152, (1536, 512), (512, 1))
    assert_size_stride(primals_153, (512, 512), (512, 1))
    assert_size_stride(primals_154, (512, ), (1, ))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (512, ), (1, ))
    assert_size_stride(primals_157, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_159, (512, ), (1, ))
    assert_size_stride(primals_160, (512, ), (1, ))
    assert_size_stride(primals_161, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_168, (512, ), (1, ))
    assert_size_stride(primals_169, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_170, (512, ), (1, ))
    assert_size_stride(primals_171, (512, ), (1, ))
    assert_size_stride(primals_172, (512, ), (1, ))
    assert_size_stride(primals_173, (1536, ), (1, ))
    assert_size_stride(primals_174, (1536, 512), (512, 1))
    assert_size_stride(primals_175, (512, 512), (512, 1))
    assert_size_stride(primals_176, (512, ), (1, ))
    assert_size_stride(primals_177, (512, ), (1, ))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (512, ), (1, ))
    assert_size_stride(primals_183, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (1536, ), (1, ))
    assert_size_stride(primals_188, (1536, 512), (512, 1))
    assert_size_stride(primals_189, (512, 512), (512, 1))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (512, ), (1, ))
    assert_size_stride(primals_193, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (512, ), (1, ))
    assert_size_stride(primals_200, (512, ), (1, ))
    assert_size_stride(primals_201, (1536, ), (1, ))
    assert_size_stride(primals_202, (1536, 512), (512, 1))
    assert_size_stride(primals_203, (512, 512), (512, 1))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (512, ), (1, ))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_214, (256, ), (1, ))
    assert_size_stride(primals_215, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, ), (1, ))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_222, (256, ), (1, ))
    assert_size_stride(primals_223, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_224, (256, ), (1, ))
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_227, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_232, (256, ), (1, ))
    assert_size_stride(primals_233, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (256, ), (1, ))
    assert_size_stride(primals_245, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (256, ), (1, ))
    assert_size_stride(primals_253, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (256, ), (1, ))
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_258, (256, ), (1, ))
    assert_size_stride(primals_259, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (256, ), (1, ))
    assert_size_stride(primals_262, (256, ), (1, ))
    assert_size_stride(primals_263, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (128, ), (1, ))
    assert_size_stride(primals_267, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_270, (128, ), (1, ))
    assert_size_stride(primals_271, (128, ), (1, ))
    assert_size_stride(primals_272, (128, ), (1, ))
    assert_size_stride(primals_273, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (128, ), (1, ))
    assert_size_stride(primals_277, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_278, (128, ), (1, ))
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_288, (128, ), (1, ))
    assert_size_stride(primals_289, (128, ), (1, ))
    assert_size_stride(primals_290, (128, ), (1, ))
    assert_size_stride(primals_291, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_292, (128, ), (1, ))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (128, ), (1, ))
    assert_size_stride(primals_295, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_300, (128, ), (1, ))
    assert_size_stride(primals_301, (128, ), (1, ))
    assert_size_stride(primals_302, (128, ), (1, ))
    assert_size_stride(primals_303, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_304, (128, ), (1, ))
    assert_size_stride(primals_305, (128, ), (1, ))
    assert_size_stride(primals_306, (128, ), (1, ))
    assert_size_stride(primals_307, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_308, (128, ), (1, ))
    assert_size_stride(primals_309, (128, ), (1, ))
    assert_size_stride(primals_310, (128, ), (1, ))
    assert_size_stride(primals_311, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (3, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_316, (3, ), (1, ))
    assert_size_stride(primals_317, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_320, (128, ), (1, ))
    assert_size_stride(primals_321, (128, ), (1, ))
    assert_size_stride(primals_322, (128, ), (1, ))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (256, ), (1, ))
    assert_size_stride(primals_327, (256, ), (1, ))
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (512, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(primals_330, (512, ), (1, ))
    assert_size_stride(primals_331, (512, ), (1, ))
    assert_size_stride(primals_332, (512, ), (1, ))
    assert_size_stride(primals_333, (512, ), (1, ))
    assert_size_stride(primals_334, (1, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(primals_335, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [h], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 32, 1, 1, 2), (64, 2, 256, 256, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 32, 1, 1, 2), (64, 2, 256, 256, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 32, 1, 1, 2), (64, 2, 256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h, input_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf1, primals_2, buf2, buf3, buf4, 256, 8192, grid=grid(256), stream=stream0)
        del primals_2
        buf5 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf6 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf8 = reinterpret_tensor(buf6, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf8, buf2, buf3, buf4, buf5, 128, 2, grid=grid(128), stream=stream0)
        buf9 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf10, buf1, buf5, buf8, primals_4, primals_5, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf12 = buf11; del buf11  # reuse
        buf13 = buf4; del buf4  # reuse
        buf14 = buf3; del buf3  # reuse
        buf15 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf12, primals_7, buf13, buf14, buf15, 256, 8192, grid=grid(256), stream=stream0)
        del primals_7
        buf16 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf19 = reinterpret_tensor(buf17, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf19, buf13, buf14, buf15, buf16, 128, 2, grid=grid(128), stream=stream0)
        buf20 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf21, buf12, buf16, buf19, primals_8, primals_9, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf23 = buf22; del buf22  # reuse
        buf24 = buf15; del buf15  # reuse
        buf25 = buf14; del buf14  # reuse
        buf26 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_9], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_3.run(buf23, primals_11, buf1, buf24, buf25, buf26, 256, 8192, grid=grid(256), stream=stream0)
        del primals_11
        buf27 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf28 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf30 = reinterpret_tensor(buf28, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf30, buf24, buf25, buf26, buf27, 128, 2, grid=grid(128), stream=stream0)
        buf31 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_4.run(buf32, buf1, buf23, buf27, buf30, primals_12, primals_13, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf34 = buf33; del buf33  # reuse
        buf35 = buf26; del buf26  # reuse
        buf36 = buf25; del buf25  # reuse
        buf37 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf34, primals_15, buf35, buf36, buf37, 256, 8192, grid=grid(256), stream=stream0)
        del primals_15
        buf38 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf39 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf41 = reinterpret_tensor(buf39, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf39  # reuse
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf41, buf35, buf36, buf37, buf38, 128, 2, grid=grid(128), stream=stream0)
        buf42 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf43 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf43, buf34, buf38, buf41, primals_16, primals_17, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf45 = empty_strided_cuda((4, 128, 65, 65), (540800, 4225, 65, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_15, input_16, x], Original ATen: [aten.add, aten.convolution, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_5.run(buf1, buf23, buf44, primals_19, buf45, 2163200, grid=grid(2163200), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_20, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf47 = buf46; del buf46  # reuse
        buf48 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf49 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf51 = reinterpret_tensor(buf49, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [x_1, input_17], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_6.run(buf47, buf51, primals_21, buf48, 128, 4096, grid=grid(128), stream=stream0)
        del primals_21
        buf52 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf53, buf47, buf48, buf51, primals_22, primals_23, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf55 = buf54; del buf54  # reuse
        buf56 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf57 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf59 = reinterpret_tensor(buf57, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf57  # reuse
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_6.run(buf55, buf59, primals_25, buf56, 128, 4096, grid=grid(128), stream=stream0)
        del primals_25
        buf60 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf61, buf55, buf56, buf59, primals_26, primals_27, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf63 = buf62; del buf62  # reuse
        buf64 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf65 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf67 = reinterpret_tensor(buf65, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf65  # reuse
        # Topologically Sorted Source Nodes: [input_23, input_25], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_8.run(buf63, buf67, primals_29, buf47, buf64, 128, 4096, grid=grid(128), stream=stream0)
        del primals_29
        buf68 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        buf69 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_9.run(buf69, buf47, buf63, buf64, buf67, primals_30, primals_31, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf71 = buf70; del buf70  # reuse
        buf72 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf73 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf75 = reinterpret_tensor(buf73, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_6.run(buf71, buf75, primals_33, buf72, 128, 4096, grid=grid(128), stream=stream0)
        del primals_33
        buf76 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf77, buf71, buf72, buf75, primals_34, primals_35, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf79 = empty_strided_cuda((4, 128, 33, 33), (139392, 1089, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_31, input_32, x_2], Original ATen: [aten.add, aten.convolution, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_10.run(buf47, buf63, buf78, primals_37, buf79, 557568, grid=grid(557568), stream=stream0)
        del primals_37
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_38, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf83 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf85 = reinterpret_tensor(buf83, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_3, input_33], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_11.run(buf81, buf85, primals_39, buf82, 128, 1024, grid=grid(128), stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf81, primals_48, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf86 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_12.run(buf87, buf81, buf82, buf85, primals_40, primals_41, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf89 = buf88; del buf88  # reuse
        buf90 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf91 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf93 = reinterpret_tensor(buf91, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [input_35, input_36], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_13.run(buf89, buf93, primals_43, buf90, 128, 2048, grid=grid(128), stream=stream0)
        del primals_43
        buf94 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf95, buf89, buf90, buf93, primals_44, primals_45, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf98 = buf97; del buf97  # reuse
        buf99 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf100 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf102 = reinterpret_tensor(buf100, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [input_39, x_4, input_40, input_41], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_15.run(buf98, buf102, primals_49, buf96, primals_47, buf99, 128, 2048, grid=grid(128), stream=stream0)
        del primals_47
        del primals_49
        buf103 = buf96; del buf96  # reuse
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf104, buf98, buf99, buf102, primals_50, primals_51, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf106 = buf105; del buf105  # reuse
        buf107 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf108 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf110 = reinterpret_tensor(buf108, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_13.run(buf106, buf110, primals_53, buf107, 128, 2048, grid=grid(128), stream=stream0)
        del primals_53
        buf111 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf112, buf106, buf107, buf110, primals_54, primals_55, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf114 = empty_strided_cuda((4, 256, 17, 17), (73984, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47, input_48, x_5], Original ATen: [aten.convolution, aten.add, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_16.run(buf98, buf113, primals_57, buf114, 295936, grid=grid(295936), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_58, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf116 = buf115; del buf115  # reuse
        buf117 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf118 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf120 = reinterpret_tensor(buf118, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_6, input_49], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_17.run(buf116, buf120, primals_59, buf117, 128, 512, grid=grid(128), stream=stream0)
        del primals_59
        buf121 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf122, buf116, buf117, buf120, primals_60, primals_61, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_62, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf124 = buf123; del buf123  # reuse
        buf125 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf126 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf128 = reinterpret_tensor(buf126, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_17.run(buf124, buf128, primals_63, buf125, 128, 512, grid=grid(128), stream=stream0)
        del primals_63
        buf129 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf130, buf124, buf125, buf128, primals_64, primals_65, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_55], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf132 = buf131; del buf131  # reuse
        buf133 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf134 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf136 = reinterpret_tensor(buf134, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [input_55, input_57], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_19.run(buf132, buf136, primals_67, buf116, buf133, 128, 512, grid=grid(128), stream=stream0)
        del primals_67
        buf137 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf138 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_20.run(buf138, buf116, buf132, buf133, buf136, primals_68, primals_69, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_70, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf140 = buf139; del buf139  # reuse
        buf141 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf142 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf144 = reinterpret_tensor(buf142, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf142  # reuse
        # Topologically Sorted Source Nodes: [input_59, input_60], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_17.run(buf140, buf144, primals_71, buf141, 128, 512, grid=grid(128), stream=stream0)
        del primals_71
        buf145 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf146 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf146, buf140, buf141, buf144, primals_72, primals_73, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf148 = empty_strided_cuda((4, 256, 9, 9), (20736, 81, 9, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, input_63, input_64, x_7], Original ATen: [aten.add, aten.convolution, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_constant_pad_nd_convolution_21.run(buf116, buf132, buf147, primals_75, buf148, 82944, grid=grid(82944), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_76, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 256, 4, 4), (4096, 16, 4, 1))
        buf150 = buf149; del buf149  # reuse
        buf151 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf152 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf154 = reinterpret_tensor(buf152, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_8, input_65], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_22.run(buf150, buf154, primals_77, buf151, 128, 128, grid=grid(128), stream=stream0)
        del primals_77
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf150, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf155 = empty_strided_cuda((4, 256, 4, 4), (4096, 16, 4, 1), torch.float32)
        buf156 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [input_65, input_66], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_23.run(buf156, buf150, buf151, buf154, primals_78, primals_79, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf158 = buf157; del buf157  # reuse
        buf159 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf160 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf162 = reinterpret_tensor(buf160, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf160  # reuse
        # Topologically Sorted Source Nodes: [input_67, input_68], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf158, buf162, primals_81, buf159, 128, 256, grid=grid(128), stream=stream0)
        del primals_81
        buf163 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf164 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf164, buf158, buf159, buf162, primals_82, primals_83, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf167 = buf166; del buf166  # reuse
        buf168 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf169 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf171 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, x_9, input_72, group_norm_18], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_26.run(buf167, primals_87, buf165, primals_85, buf168, buf169, buf171, 128, 256, grid=grid(128), stream=stream0)
        del primals_85
        del primals_87
        buf172 = reinterpret_tensor(buf165, (16, 4, 512), (2048, 512, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_27.run(buf167, buf168, buf169, primals_88, primals_89, buf172, 16, 2048, grid=grid(16, 2048), stream=stream0)
        del primals_89
        buf173 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (64, 512), (512, 1), 0), reinterpret_tensor(primals_91, (512, 1536), (1, 512), 0), out=buf173)
        buf174 = empty_strided_cuda((3, 16, 4, 512), (32768, 2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf173, primals_90, buf174, 98304, grid=grid(98304), stream=stream0)
        del primals_90
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf175 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf174, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf174, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf174, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), None, True)
        buf176 = buf175[0]
        buf177 = buf175[1]
        buf178 = buf175[2]
        buf179 = buf175[3]
        del buf175
        buf180 = empty_strided_cuda((16, 4, 1, 512), (2048, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf176, buf180, 32768, grid=grid(32768), stream=stream0)
        buf181 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf180, (64, 512), (512, 1), 0), reinterpret_tensor(primals_92, (512, 512), (1, 512), 0), out=buf181)
        buf182 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_30.run(buf167, buf181, primals_93, buf182, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_93
        buf183 = reinterpret_tensor(buf169, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf169  # reuse
        buf184 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf186 = reinterpret_tensor(buf184, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf184  # reuse
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf186, buf182, buf183, 128, 256, grid=grid(128), stream=stream0)
        buf187 = reinterpret_tensor(buf181, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf181  # reuse
        buf188 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf188, buf182, buf183, buf186, primals_94, primals_95, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf190 = buf189; del buf189  # reuse
        buf191 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf192 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf194 = reinterpret_tensor(buf192, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [input_75, input_76], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf190, buf194, primals_97, buf191, 128, 256, grid=grid(128), stream=stream0)
        del primals_97
        buf195 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf196 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf196, buf190, buf191, buf194, primals_98, primals_99, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf198 = buf197; del buf197  # reuse
        buf199 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf200 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf202 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_79, input_80, group_norm_21], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_32.run(buf198, buf182, primals_101, buf199, buf200, buf202, 128, 256, grid=grid(128), stream=stream0)
        del primals_101
        buf203 = empty_strided_cuda((16, 4, 512), (2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_27.run(buf198, buf199, buf200, primals_102, primals_103, buf203, 16, 2048, grid=grid(16, 2048), stream=stream0)
        del primals_103
        buf204 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf203, (64, 512), (512, 1), 0), reinterpret_tensor(primals_105, (512, 1536), (1, 512), 0), out=buf204)
        buf205 = empty_strided_cuda((3, 16, 4, 512), (32768, 2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf204, primals_104, buf205, 98304, grid=grid(98304), stream=stream0)
        del primals_104
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf206 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf205, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf205, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf205, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), None, True)
        buf207 = buf206[0]
        buf208 = buf206[1]
        buf209 = buf206[2]
        buf210 = buf206[3]
        del buf206
        buf211 = empty_strided_cuda((16, 4, 1, 512), (2048, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf207, buf211, 32768, grid=grid(32768), stream=stream0)
        buf212 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf211, (64, 512), (512, 1), 0), reinterpret_tensor(primals_106, (512, 512), (1, 512), 0), out=buf212)
        buf213 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_30.run(buf198, buf212, primals_107, buf213, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_107
        buf214 = reinterpret_tensor(buf200, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf200  # reuse
        buf215 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf217 = reinterpret_tensor(buf215, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf217, buf213, buf214, 128, 256, grid=grid(128), stream=stream0)
        buf218 = reinterpret_tensor(buf212, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf212  # reuse
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [input_81, input_82], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf219, buf213, buf214, buf217, primals_108, primals_109, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf221 = buf220; del buf220  # reuse
        buf222 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf223 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf225 = reinterpret_tensor(buf223, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf223  # reuse
        # Topologically Sorted Source Nodes: [input_83, input_84], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf221, buf225, primals_111, buf222, 128, 256, grid=grid(128), stream=stream0)
        del primals_111
        buf226 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf227 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf227, buf221, buf222, buf225, primals_112, primals_113, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_87], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf229 = buf228; del buf228  # reuse
        buf230 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf231 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf233 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_87, input_88, group_norm_24], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_32.run(buf229, buf213, primals_115, buf230, buf231, buf233, 128, 256, grid=grid(128), stream=stream0)
        del primals_115
        buf234 = empty_strided_cuda((16, 4, 512), (2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_27.run(buf229, buf230, buf231, primals_116, primals_117, buf234, 16, 2048, grid=grid(16, 2048), stream=stream0)
        del primals_117
        buf235 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (64, 512), (512, 1), 0), reinterpret_tensor(primals_119, (512, 1536), (1, 512), 0), out=buf235)
        buf236 = empty_strided_cuda((3, 16, 4, 512), (32768, 2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf235, primals_118, buf236, 98304, grid=grid(98304), stream=stream0)
        del primals_118
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf237 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf236, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf236, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf236, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), None, True)
        buf238 = buf237[0]
        buf239 = buf237[1]
        buf240 = buf237[2]
        buf241 = buf237[3]
        del buf237
        buf272 = empty_strided_cuda((4, 6), (6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dist], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_33.run(primals_136, buf272, 24, grid=grid(24), stream=stream0)
        buf242 = empty_strided_cuda((16, 4, 1, 512), (2048, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf238, buf242, 32768, grid=grid(32768), stream=stream0)
        buf243 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf242, (64, 512), (512, 1), 0), reinterpret_tensor(primals_120, (512, 512), (1, 512), 0), out=buf243)
        buf244 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_30.run(buf229, buf243, primals_121, buf244, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_121
        buf245 = reinterpret_tensor(buf231, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf231  # reuse
        buf246 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf248 = reinterpret_tensor(buf246, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf246  # reuse
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf248, buf244, buf245, 128, 256, grid=grid(128), stream=stream0)
        buf249 = reinterpret_tensor(buf243, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf243  # reuse
        buf250 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf250, buf244, buf245, buf248, primals_122, primals_123, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf252 = buf251; del buf251  # reuse
        buf253 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf254 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf256 = reinterpret_tensor(buf254, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf254  # reuse
        # Topologically Sorted Source Nodes: [input_91, input_92], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf252, buf256, primals_125, buf253, 128, 256, grid=grid(128), stream=stream0)
        del primals_125
        buf257 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf258, buf252, buf253, buf256, primals_126, primals_127, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_95], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_128, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf260 = buf259; del buf259  # reuse
        buf261 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf262 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf264 = reinterpret_tensor(buf262, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [input_95, input_96, input_97], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_34.run(buf260, buf264, buf244, primals_129, buf261, 128, 256, grid=grid(128), stream=stream0)
        del primals_129
        buf265 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf266 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [input_97, input_98], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf266, buf260, buf261, buf264, primals_130, primals_131, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_132, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 4, 4, 4), (64, 16, 4, 1))
        buf268 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_35.run(buf268, primals_133, 256, grid=grid(256), stream=stream0)
        del primals_133
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 4, 4, 4), (64, 16, 4, 1))
        buf270 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_35.run(buf270, primals_135, 256, grid=grid(256), stream=stream0)
        del primals_135
        buf271 = empty_strided_cuda((64, 6), (6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dist], Original ATen: [aten._euclidean_dist]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_36.run(buf270, buf271, 384, grid=grid(384), stream=stream0)
        buf273 = reinterpret_tensor(buf37, (64, 4), (4, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [dist], Original ATen: [aten._euclidean_dist]
        extern_kernels.mm(buf271, reinterpret_tensor(buf272, (6, 4), (1, 6), 0), out=buf273)
        del buf271
        del buf272
        buf274 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [dist, argmin], Original ATen: [aten._euclidean_dist, aten.argmin]
        stream0 = get_raw_stream(0)
        triton_poi_fused__euclidean_dist_argmin_37.run(buf273, buf274, 64, grid=grid(64), stream=stream0)
        buf275 = reinterpret_tensor(buf273, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [embeddings], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_38.run(buf274, primals_136, buf275, 256, grid=grid(256), stream=stream0)
        del primals_136
        buf277 = reinterpret_tensor(buf36, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [sub, x_q_1], Original ATen: [aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sub_39.run(buf270, buf275, buf277, 16, 16, grid=grid(16, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [h_4], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 4, 4, 4), (64, 16, 4, 1))
        buf279 = buf278; del buf278  # reuse
        # Topologically Sorted Source Nodes: [h_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_35.run(buf279, primals_138, 256, grid=grid(256), stream=stream0)
        del primals_138
        # Topologically Sorted Source Nodes: [h_5], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf281 = buf280; del buf280  # reuse
        buf282 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf283 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf285 = reinterpret_tensor(buf283, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf283  # reuse
        # Topologically Sorted Source Nodes: [h_5, input_101], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf281, buf285, primals_140, buf282, 128, 256, grid=grid(128), stream=stream0)
        del primals_140
        buf286 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf287 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [input_101, input_102], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf287, buf281, buf282, buf285, primals_141, primals_142, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf289 = buf288; del buf288  # reuse
        buf290 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf291 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf293 = reinterpret_tensor(buf291, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [input_103, input_104], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf289, buf293, primals_144, buf290, 128, 256, grid=grid(128), stream=stream0)
        del primals_144
        buf294 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf295 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf295, buf289, buf290, buf293, primals_145, primals_146, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf297 = buf296; del buf296  # reuse
        buf298 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf299 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf301 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_107, group_norm_30], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_40.run(buf297, primals_148, buf281, buf298, buf299, buf301, 128, 256, grid=grid(128), stream=stream0)
        del primals_148
        buf302 = empty_strided_cuda((16, 4, 512), (2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_41.run(buf281, buf297, buf298, buf299, primals_149, primals_150, buf302, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_150
        buf303 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (64, 512), (512, 1), 0), reinterpret_tensor(primals_152, (512, 1536), (1, 512), 0), out=buf303)
        buf304 = empty_strided_cuda((3, 16, 4, 512), (32768, 2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf303, primals_151, buf304, 98304, grid=grid(98304), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf305 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf304, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf304, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf304, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), None, True)
        buf306 = buf305[0]
        buf307 = buf305[1]
        buf308 = buf305[2]
        buf309 = buf305[3]
        del buf305
        buf310 = empty_strided_cuda((16, 4, 1, 512), (2048, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf306, buf310, 32768, grid=grid(32768), stream=stream0)
        buf311 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf310, (64, 512), (512, 1), 0), reinterpret_tensor(primals_153, (512, 512), (1, 512), 0), out=buf311)
        buf312 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_108, out_11], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_42.run(buf281, buf297, buf311, primals_154, buf312, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_154
        buf313 = reinterpret_tensor(buf299, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf299  # reuse
        buf314 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf316 = reinterpret_tensor(buf314, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf314  # reuse
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf316, buf312, buf313, 128, 256, grid=grid(128), stream=stream0)
        buf317 = reinterpret_tensor(buf311, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf311  # reuse
        buf318 = buf317; del buf317  # reuse
        # Topologically Sorted Source Nodes: [input_109, input_110], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf318, buf312, buf313, buf316, primals_155, primals_156, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf320 = buf319; del buf319  # reuse
        buf321 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf322 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf324 = reinterpret_tensor(buf322, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf322  # reuse
        # Topologically Sorted Source Nodes: [input_111, input_112], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf320, buf324, primals_158, buf321, 128, 256, grid=grid(128), stream=stream0)
        del primals_158
        buf325 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf326 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [input_112, input_113], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf326, buf320, buf321, buf324, primals_159, primals_160, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf328 = buf327; del buf327  # reuse
        buf329 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf330 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf332 = reinterpret_tensor(buf330, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf330  # reuse
        # Topologically Sorted Source Nodes: [input_115, input_116, input_117], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_34.run(buf328, buf332, buf312, primals_162, buf329, 128, 256, grid=grid(128), stream=stream0)
        del primals_162
        buf333 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf334 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [input_117, input_118], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf334, buf328, buf329, buf332, primals_163, primals_164, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_119], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf336 = buf335; del buf335  # reuse
        buf337 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf338 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf340 = reinterpret_tensor(buf338, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf338  # reuse
        # Topologically Sorted Source Nodes: [input_119, input_120], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf336, buf340, primals_166, buf337, 128, 256, grid=grid(128), stream=stream0)
        del primals_166
        buf341 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf342 = buf341; del buf341  # reuse
        # Topologically Sorted Source Nodes: [input_120, input_121], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf342, buf336, buf337, buf340, primals_167, primals_168, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_123], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, primals_169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf344 = buf343; del buf343  # reuse
        buf345 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf346 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf348 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_123, input_124, group_norm_35], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_32.run(buf344, buf328, primals_170, buf345, buf346, buf348, 128, 256, grid=grid(128), stream=stream0)
        del primals_170
        buf349 = empty_strided_cuda((16, 4, 512), (2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_27.run(buf344, buf345, buf346, primals_171, primals_172, buf349, 16, 2048, grid=grid(16, 2048), stream=stream0)
        del primals_172
        buf350 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (64, 512), (512, 1), 0), reinterpret_tensor(primals_174, (512, 1536), (1, 512), 0), out=buf350)
        buf351 = empty_strided_cuda((3, 16, 4, 512), (32768, 2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf350, primals_173, buf351, 98304, grid=grid(98304), stream=stream0)
        del primals_173
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf352 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf351, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf351, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf351, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), None, True)
        buf353 = buf352[0]
        buf354 = buf352[1]
        buf355 = buf352[2]
        buf356 = buf352[3]
        del buf352
        buf357 = empty_strided_cuda((16, 4, 1, 512), (2048, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf353, buf357, 32768, grid=grid(32768), stream=stream0)
        buf358 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf357, (64, 512), (512, 1), 0), reinterpret_tensor(primals_175, (512, 512), (1, 512), 0), out=buf358)
        buf359 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_30.run(buf344, buf358, primals_176, buf359, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_176
        buf360 = reinterpret_tensor(buf346, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf346  # reuse
        buf361 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf363 = reinterpret_tensor(buf361, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf361  # reuse
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf363, buf359, buf360, 128, 256, grid=grid(128), stream=stream0)
        buf364 = reinterpret_tensor(buf358, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf358  # reuse
        buf365 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [input_125, input_126], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf365, buf359, buf360, buf363, primals_177, primals_178, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_127], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_179, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf367 = buf366; del buf366  # reuse
        buf368 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf369 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf371 = reinterpret_tensor(buf369, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf369  # reuse
        # Topologically Sorted Source Nodes: [input_127, input_128], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf367, buf371, primals_180, buf368, 128, 256, grid=grid(128), stream=stream0)
        del primals_180
        buf372 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf373 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf373, buf367, buf368, buf371, primals_181, primals_182, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf373, primals_183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf375 = buf374; del buf374  # reuse
        buf376 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf377 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf379 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_131, input_132, group_norm_38], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_32.run(buf375, buf359, primals_184, buf376, buf377, buf379, 128, 256, grid=grid(128), stream=stream0)
        del primals_184
        buf380 = empty_strided_cuda((16, 4, 512), (2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_27.run(buf375, buf376, buf377, primals_185, primals_186, buf380, 16, 2048, grid=grid(16, 2048), stream=stream0)
        del primals_186
        buf381 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf380, (64, 512), (512, 1), 0), reinterpret_tensor(primals_188, (512, 1536), (1, 512), 0), out=buf381)
        buf382 = empty_strided_cuda((3, 16, 4, 512), (32768, 2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf381, primals_187, buf382, 98304, grid=grid(98304), stream=stream0)
        del primals_187
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf383 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf382, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf382, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf382, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), None, True)
        buf384 = buf383[0]
        buf385 = buf383[1]
        buf386 = buf383[2]
        buf387 = buf383[3]
        del buf383
        buf388 = empty_strided_cuda((16, 4, 1, 512), (2048, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf384, buf388, 32768, grid=grid(32768), stream=stream0)
        buf389 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf388, (64, 512), (512, 1), 0), reinterpret_tensor(primals_189, (512, 512), (1, 512), 0), out=buf389)
        buf390 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_17], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_30.run(buf375, buf389, primals_190, buf390, 2048, 16, grid=grid(2048, 16), stream=stream0)
        del primals_190
        buf391 = reinterpret_tensor(buf377, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf377  # reuse
        buf392 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf394 = reinterpret_tensor(buf392, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf392  # reuse
        # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_31.run(buf394, buf390, buf391, 128, 256, grid=grid(128), stream=stream0)
        buf395 = reinterpret_tensor(buf389, (4, 512, 4, 4), (8192, 16, 4, 1), 0); del buf389  # reuse
        buf396 = buf395; del buf395  # reuse
        # Topologically Sorted Source Nodes: [input_133, input_134], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf396, buf390, buf391, buf394, primals_191, primals_192, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_135], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, primals_193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf398 = buf397; del buf397  # reuse
        buf399 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf400 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf402 = reinterpret_tensor(buf400, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [input_135, input_136], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_24.run(buf398, buf402, primals_194, buf399, 128, 256, grid=grid(128), stream=stream0)
        del primals_194
        buf403 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf404 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [input_136, input_137], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_25.run(buf404, buf398, buf399, buf402, primals_195, primals_196, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf404, primals_197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf406 = buf405; del buf405  # reuse
        buf407 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf408 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf410 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_139, input_140, group_norm_41], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_32.run(buf406, buf390, primals_198, buf407, buf408, buf410, 128, 256, grid=grid(128), stream=stream0)
        del primals_198
        buf411 = empty_strided_cuda((16, 4, 512), (2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_27.run(buf406, buf407, buf408, primals_199, primals_200, buf411, 16, 2048, grid=grid(16, 2048), stream=stream0)
        del primals_200
        buf412 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (64, 512), (512, 1), 0), reinterpret_tensor(primals_202, (512, 1536), (1, 512), 0), out=buf412)
        buf413 = empty_strided_cuda((3, 16, 4, 512), (32768, 2048, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_28.run(buf412, primals_201, buf413, 98304, grid=grid(98304), stream=stream0)
        del buf412
        del primals_201
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf414 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf413, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf413, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf413, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), None, True)
        buf415 = buf414[0]
        buf416 = buf414[1]
        buf417 = buf414[2]
        buf418 = buf414[3]
        del buf414
        buf421 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_43.run(buf421, 8, grid=grid(8), stream=stream0)
        buf473 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_44.run(buf473, 16, grid=grid(16), stream=stream0)
        buf524 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_45.run(buf524, 32, grid=grid(32), stream=stream0)
        buf576 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_46.run(buf576, 64, grid=grid(64), stream=stream0)
        buf419 = empty_strided_cuda((16, 4, 1, 512), (2048, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_29.run(buf415, buf419, 32768, grid=grid(32768), stream=stream0)
        buf420 = empty_strided_cuda((64, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf419, (64, 512), (512, 1), 0), reinterpret_tensor(primals_203, (512, 512), (1, 512), 0), out=buf420)
        buf422 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_20, x_10], Original ATen: [aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_47.run(buf421, buf406, buf420, primals_204, buf422, 131072, grid=grid(131072), stream=stream0)
        del buf420
        del primals_204
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf424 = buf423; del buf423  # reuse
        buf425 = reinterpret_tensor(buf408, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf408  # reuse
        buf426 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf428 = reinterpret_tensor(buf426, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf426  # reuse
        # Topologically Sorted Source Nodes: [input_141, input_142], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_48.run(buf424, buf428, primals_206, buf425, 128, 1024, grid=grid(128), stream=stream0)
        del primals_206
        buf429 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf430 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [input_142, input_143], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_49.run(buf430, buf424, buf425, buf428, primals_207, primals_208, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf431, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf432 = buf431; del buf431  # reuse
        buf433 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf434 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf436 = reinterpret_tensor(buf434, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [input_144, input_145], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_17.run(buf432, buf436, primals_210, buf433, 128, 512, grid=grid(128), stream=stream0)
        del primals_210
        buf437 = buf147; del buf147  # reuse
        buf438 = buf437; del buf437  # reuse
        # Topologically Sorted Source Nodes: [input_145, input_146], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf438, buf432, buf433, buf436, primals_211, primals_212, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_148], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 256, 8, 8), (16384, 64, 8, 1))
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf424, primals_215, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf441 = buf440; del buf440  # reuse
        buf442 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf443 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf445 = reinterpret_tensor(buf443, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf443  # reuse
        # Topologically Sorted Source Nodes: [input_148, x_11, input_149, input_150], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_50.run(buf441, buf445, primals_216, buf439, primals_214, buf442, 128, 512, grid=grid(128), stream=stream0)
        del primals_214
        del primals_216
        buf446 = buf439; del buf439  # reuse
        buf447 = buf446; del buf446  # reuse
        # Topologically Sorted Source Nodes: [input_150, input_151], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf447, buf441, buf442, buf445, primals_217, primals_218, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_152], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf449 = buf448; del buf448  # reuse
        buf450 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf451 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf453 = reinterpret_tensor(buf451, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [input_152, input_153], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_17.run(buf449, buf453, primals_220, buf450, 128, 512, grid=grid(128), stream=stream0)
        del primals_220
        buf454 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf455 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [input_153, input_154], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf455, buf449, buf450, buf453, primals_221, primals_222, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf457 = buf456; del buf456  # reuse
        buf458 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf459 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf461 = reinterpret_tensor(buf459, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf459  # reuse
        # Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_convolution_native_group_norm_51.run(buf457, buf461, buf441, primals_224, buf458, 128, 512, grid=grid(128), stream=stream0)
        del primals_224
        buf462 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf463 = buf462; del buf462  # reuse
        # Topologically Sorted Source Nodes: [input_158, input_159], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf463, buf457, buf458, buf461, primals_225, primals_226, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf465 = buf464; del buf464  # reuse
        buf466 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf467 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf469 = reinterpret_tensor(buf467, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf467  # reuse
        # Topologically Sorted Source Nodes: [input_160, input_161], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_convolution_native_group_norm_17.run(buf465, buf469, primals_228, buf466, 128, 512, grid=grid(128), stream=stream0)
        del primals_228
        buf470 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf471 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [input_161, input_162], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_18.run(buf471, buf465, buf466, buf469, primals_229, primals_230, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_164], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf471, primals_231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf474 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [input_164, input_165, x_12], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_52.run(buf473, buf457, buf472, primals_232, buf474, 262144, grid=grid(262144), stream=stream0)
        del primals_232
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, primals_233, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf476 = buf475; del buf475  # reuse
        buf477 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf478 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf480 = reinterpret_tensor(buf478, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf478  # reuse
        # Topologically Sorted Source Nodes: [input_166, input_167], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_13.run(buf476, buf480, primals_234, buf477, 128, 2048, grid=grid(128), stream=stream0)
        del primals_234
        buf481 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf482 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [input_167, input_168], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf482, buf476, buf477, buf480, primals_235, primals_236, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf484 = buf483; del buf483  # reuse
        buf485 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf486 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf488 = reinterpret_tensor(buf486, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf486  # reuse
        # Topologically Sorted Source Nodes: [input_169, input_170], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_13.run(buf484, buf488, primals_238, buf485, 128, 2048, grid=grid(128), stream=stream0)
        del primals_238
        buf489 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf490 = buf489; del buf489  # reuse
        # Topologically Sorted Source Nodes: [input_170, input_171], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf490, buf484, buf485, buf488, primals_239, primals_240, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_173], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, primals_241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf492 = buf491; del buf491  # reuse
        buf493 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf494 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf496 = reinterpret_tensor(buf494, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [input_173, input_175], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_53.run(buf492, buf496, primals_242, buf476, buf493, 128, 2048, grid=grid(128), stream=stream0)
        del primals_242
        buf497 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf498 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [input_175, input_176], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_54.run(buf498, buf476, buf492, buf493, buf496, primals_243, primals_244, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_177], Original ATen: [aten.convolution]
        buf499 = extern_kernels.convolution(buf498, primals_245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf499, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf500 = buf499; del buf499  # reuse
        buf501 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf502 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf504 = reinterpret_tensor(buf502, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf502  # reuse
        # Topologically Sorted Source Nodes: [input_177, input_178], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_13.run(buf500, buf504, primals_246, buf501, 128, 2048, grid=grid(128), stream=stream0)
        del primals_246
        buf505 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf506 = buf505; del buf505  # reuse
        # Topologically Sorted Source Nodes: [input_178, input_179], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf506, buf500, buf501, buf504, primals_247, primals_248, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_181], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(buf506, primals_249, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf507, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf508 = buf507; del buf507  # reuse
        buf509 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf510 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf512 = reinterpret_tensor(buf510, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf510  # reuse
        # Topologically Sorted Source Nodes: [input_174, input_181, input_182, input_183], Original ATen: [aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_55.run(buf508, buf512, buf476, buf492, primals_250, buf509, 128, 2048, grid=grid(128), stream=stream0)
        del primals_250
        buf513 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf514 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [input_183, input_184], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf514, buf508, buf509, buf512, primals_251, primals_252, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_185], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_253, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf516 = buf515; del buf515  # reuse
        buf517 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf518 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf520 = reinterpret_tensor(buf518, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf518  # reuse
        # Topologically Sorted Source Nodes: [input_185, input_186], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_13.run(buf516, buf520, primals_254, buf517, 128, 2048, grid=grid(128), stream=stream0)
        del primals_254
        buf521 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf522 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [input_186, input_187], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_14.run(buf522, buf516, buf517, buf520, primals_255, primals_256, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_189], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf523, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf525 = empty_strided_cuda((4, 256, 32, 32), (262144, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_189, input_190, x_13], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_56.run(buf524, buf508, buf523, primals_258, buf525, 1048576, grid=grid(1048576), stream=stream0)
        del buf523
        del primals_258
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 256, 32, 32), (262144, 1024, 32, 1))
        buf527 = buf526; del buf526  # reuse
        buf528 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf529 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf531 = reinterpret_tensor(buf529, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [input_191, input_192], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_57.run(buf527, buf531, primals_260, buf528, 128, 8192, grid=grid(128), stream=stream0)
        del primals_260
        buf532 = empty_strided_cuda((4, 256, 32, 32), (262144, 1024, 32, 1), torch.float32)
        buf533 = buf532; del buf532  # reuse
        # Topologically Sorted Source Nodes: [input_192, input_193], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_58.run(buf533, buf527, buf528, buf531, primals_261, primals_262, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_194], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_263, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf535 = buf534; del buf534  # reuse
        buf536 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf537 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf539 = reinterpret_tensor(buf537, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf537  # reuse
        # Topologically Sorted Source Nodes: [input_194, input_195], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_6.run(buf535, buf539, primals_264, buf536, 128, 4096, grid=grid(128), stream=stream0)
        del primals_264
        buf540 = buf78; del buf78  # reuse
        buf541 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [input_195, input_196], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf541, buf535, buf536, buf539, primals_265, primals_266, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_198], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (4, 128, 32, 32), (131072, 1024, 32, 1))
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf543 = extern_kernels.convolution(buf527, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf544 = buf543; del buf543  # reuse
        buf545 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf546 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf548 = reinterpret_tensor(buf546, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf546  # reuse
        # Topologically Sorted Source Nodes: [input_198, x_14, input_199, input_200], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_59.run(buf544, buf548, primals_270, buf542, primals_268, buf545, 128, 4096, grid=grid(128), stream=stream0)
        del primals_268
        del primals_270
        buf549 = buf542; del buf542  # reuse
        buf550 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [input_200, input_201], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf550, buf544, buf545, buf548, primals_271, primals_272, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_202], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_273, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf552 = buf551; del buf551  # reuse
        buf553 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf554 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf556 = reinterpret_tensor(buf554, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf554  # reuse
        # Topologically Sorted Source Nodes: [input_202, input_203], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_6.run(buf552, buf556, primals_274, buf553, 128, 4096, grid=grid(128), stream=stream0)
        del primals_274
        buf557 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        buf558 = buf557; del buf557  # reuse
        # Topologically Sorted Source Nodes: [input_203, input_204], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf558, buf552, buf553, buf556, primals_275, primals_276, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf559 = extern_kernels.convolution(buf558, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf559, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf560 = buf559; del buf559  # reuse
        buf561 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf562 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf564 = reinterpret_tensor(buf562, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf562  # reuse
        # Topologically Sorted Source Nodes: [input_206, input_207, input_208], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_60.run(buf560, buf564, buf544, primals_278, buf561, 128, 4096, grid=grid(128), stream=stream0)
        del primals_278
        buf565 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        buf566 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [input_208, input_209], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf566, buf560, buf561, buf564, primals_279, primals_280, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_210], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, primals_281, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf568 = buf567; del buf567  # reuse
        buf569 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf570 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf572 = reinterpret_tensor(buf570, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf570  # reuse
        # Topologically Sorted Source Nodes: [input_210, input_211], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_6.run(buf568, buf572, primals_282, buf569, 128, 4096, grid=grid(128), stream=stream0)
        del primals_282
        buf573 = empty_strided_cuda((4, 128, 32, 32), (131072, 1024, 32, 1), torch.float32)
        buf574 = buf573; del buf573  # reuse
        # Topologically Sorted Source Nodes: [input_211, input_212], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_7.run(buf574, buf568, buf569, buf572, primals_283, primals_284, 524288, grid=grid(524288), stream=stream0)
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf574, primals_285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf575, (4, 128, 32, 32), (131072, 1024, 32, 1))
        buf577 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_214, input_215, x_15], Original ATen: [aten.convolution, aten.add, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_convolution_61.run(buf576, buf560, buf575, primals_286, buf577, 2097152, grid=grid(2097152), stream=stream0)
        del buf575
        del primals_286
        # Topologically Sorted Source Nodes: [input_216], Original ATen: [aten.convolution]
        buf578 = extern_kernels.convolution(buf577, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf579 = buf578; del buf578  # reuse
        buf580 = buf35; del buf35  # reuse
        buf581 = empty_strided_cuda((4, 32, 1, 1, 2), (64, 2, 256, 256, 1), torch.float32)
        buf582 = empty_strided_cuda((4, 32, 1, 1, 2), (64, 2, 256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_216, input_217], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf579, primals_288, buf580, buf581, buf582, 256, 8192, grid=grid(256), stream=stream0)
        del primals_288
        buf583 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf584 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf586 = reinterpret_tensor(buf584, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf584  # reuse
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf586, buf580, buf581, buf582, buf583, 128, 2, grid=grid(128), stream=stream0)
        buf587 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf588 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [input_217, input_218], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf588, buf579, buf583, buf586, primals_289, primals_290, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_219], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, primals_291, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf589, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf590 = buf589; del buf589  # reuse
        buf591 = buf582; del buf582  # reuse
        buf592 = buf581; del buf581  # reuse
        buf593 = buf580; del buf580  # reuse
        # Topologically Sorted Source Nodes: [input_219, input_220], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf590, primals_292, buf591, buf592, buf593, 256, 8192, grid=grid(256), stream=stream0)
        del primals_292
        buf594 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf595 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf597 = reinterpret_tensor(buf595, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf595  # reuse
        # Topologically Sorted Source Nodes: [input_220], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf597, buf591, buf592, buf593, buf594, 128, 2, grid=grid(128), stream=stream0)
        buf598 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf599 = buf598; del buf598  # reuse
        # Topologically Sorted Source Nodes: [input_220, input_221], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf599, buf590, buf594, buf597, primals_293, primals_294, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_223], Original ATen: [aten.convolution]
        buf600 = extern_kernels.convolution(buf599, primals_295, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf601 = buf600; del buf600  # reuse
        buf602 = buf593; del buf593  # reuse
        buf603 = buf592; del buf592  # reuse
        buf604 = buf591; del buf591  # reuse
        # Topologically Sorted Source Nodes: [input_223, input_225], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_3.run(buf601, primals_296, buf579, buf602, buf603, buf604, 256, 8192, grid=grid(256), stream=stream0)
        del primals_296
        buf605 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf606 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf608 = reinterpret_tensor(buf606, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf606  # reuse
        # Topologically Sorted Source Nodes: [input_225], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf608, buf602, buf603, buf604, buf605, 128, 2, grid=grid(128), stream=stream0)
        buf609 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf610 = buf609; del buf609  # reuse
        # Topologically Sorted Source Nodes: [input_225, input_226], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_4.run(buf610, buf579, buf601, buf605, buf608, primals_297, primals_298, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_227], Original ATen: [aten.convolution]
        buf611 = extern_kernels.convolution(buf610, primals_299, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf611, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf612 = buf611; del buf611  # reuse
        buf613 = buf604; del buf604  # reuse
        buf614 = buf603; del buf603  # reuse
        buf615 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [input_227, input_228], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf612, primals_300, buf613, buf614, buf615, 256, 8192, grid=grid(256), stream=stream0)
        del primals_300
        buf616 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf617 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf619 = reinterpret_tensor(buf617, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf617  # reuse
        # Topologically Sorted Source Nodes: [input_228], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf619, buf613, buf614, buf615, buf616, 128, 2, grid=grid(128), stream=stream0)
        buf620 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf621 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [input_228, input_229], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf621, buf612, buf616, buf619, primals_301, primals_302, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_231], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf621, primals_303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf622, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf623 = buf622; del buf622  # reuse
        buf624 = buf615; del buf615  # reuse
        buf625 = buf614; del buf614  # reuse
        buf626 = buf613; del buf613  # reuse
        # Topologically Sorted Source Nodes: [input_224, input_231, input_232, input_233], Original ATen: [aten.add, aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_62.run(buf623, buf579, buf601, primals_304, buf624, buf625, buf626, 256, 8192, grid=grid(256), stream=stream0)
        del primals_304
        buf627 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf628 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf630 = reinterpret_tensor(buf628, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf628  # reuse
        # Topologically Sorted Source Nodes: [input_233], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf630, buf624, buf625, buf626, buf627, 128, 2, grid=grid(128), stream=stream0)
        buf631 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf632 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [input_233, input_234], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf632, buf623, buf627, buf630, primals_305, primals_306, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_235], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf634 = buf633; del buf633  # reuse
        buf635 = buf626; del buf626  # reuse
        buf636 = buf625; del buf625  # reuse
        buf637 = buf624; del buf624  # reuse
        # Topologically Sorted Source Nodes: [input_235, input_236], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_0.run(buf634, primals_308, buf635, buf636, buf637, 256, 8192, grid=grid(256), stream=stream0)
        del primals_308
        buf638 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf639 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf641 = reinterpret_tensor(buf639, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf639  # reuse
        # Topologically Sorted Source Nodes: [input_236], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf641, buf635, buf636, buf637, buf638, 128, 2, grid=grid(128), stream=stream0)
        buf642 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf643 = buf642; del buf642  # reuse
        # Topologically Sorted Source Nodes: [input_236, input_237], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf643, buf634, buf638, buf641, primals_309, primals_310, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_239], Original ATen: [aten.convolution]
        buf644 = extern_kernels.convolution(buf643, primals_311, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf644, (4, 128, 64, 64), (524288, 4096, 64, 1))
        buf645 = buf644; del buf644  # reuse
        buf646 = buf637; del buf637  # reuse
        buf647 = buf636; del buf636  # reuse
        buf648 = buf635; del buf635  # reuse
        # Topologically Sorted Source Nodes: [input_239, input_240, input_241], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_63.run(buf645, buf623, primals_312, buf646, buf647, buf648, 256, 8192, grid=grid(256), stream=stream0)
        del primals_312
        buf649 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 1, 1), torch.float32)
        buf650 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf652 = reinterpret_tensor(buf650, (4, 32, 1, 1), (32, 1, 1, 1), 0); del buf650  # reuse
        # Topologically Sorted Source Nodes: [input_241], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf652, buf646, buf647, buf648, buf649, 128, 2, grid=grid(128), stream=stream0)
        del buf646
        del buf647
        del buf648
        buf653 = empty_strided_cuda((4, 128, 64, 64), (524288, 4096, 64, 1), torch.float32)
        buf654 = buf653; del buf653  # reuse
        # Topologically Sorted Source Nodes: [input_241, input_242], Original ATen: [aten.native_group_norm, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_silu_2.run(buf654, buf645, buf649, buf652, primals_313, primals_314, 2097152, grid=grid(2097152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
        buf655 = extern_kernels.convolution(buf654, primals_315, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf655, (4, 3, 64, 64), (12288, 4096, 64, 1))
        buf656 = buf655; del buf655  # reuse
        # Topologically Sorted Source Nodes: [input_243], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_64.run(buf656, primals_316, 49152, grid=grid(49152), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [input_244], Original ATen: [aten.convolution]
        buf657 = extern_kernels.convolution(buf656, primals_317, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf658 = buf657; del buf657  # reuse
        # Topologically Sorted Source Nodes: [input_244, input_245], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_65.run(buf658, primals_318, 262144, grid=grid(262144), stream=stream0)
        del primals_318
        # Topologically Sorted Source Nodes: [input_246], Original ATen: [aten.convolution]
        buf659 = extern_kernels.convolution(buf658, primals_319, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf659, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf660 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf661 = buf660; del buf660  # reuse
        # Topologically Sorted Source Nodes: [input_247, input_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_66.run(buf661, buf659, primals_320, primals_321, primals_322, primals_323, 131072, grid=grid(131072), stream=stream0)
        del primals_323
        # Topologically Sorted Source Nodes: [input_249], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, primals_324, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf663 = buf472; del buf472  # reuse
        buf664 = buf663; del buf663  # reuse
        # Topologically Sorted Source Nodes: [input_250, input_251], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_67.run(buf664, buf662, primals_325, primals_326, primals_327, primals_328, 65536, grid=grid(65536), stream=stream0)
        del primals_328
        # Topologically Sorted Source Nodes: [input_252], Original ATen: [aten.convolution]
        buf665 = extern_kernels.convolution(buf664, primals_329, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf665, (4, 512, 7, 7), (25088, 49, 7, 1))
        buf666 = empty_strided_cuda((4, 512, 7, 7), (25088, 49, 7, 1), torch.float32)
        buf667 = buf666; del buf666  # reuse
        # Topologically Sorted Source Nodes: [input_253, input_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_68.run(buf667, buf665, primals_330, primals_331, primals_332, primals_333, 100352, grid=grid(100352), stream=stream0)
        del primals_333
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
        buf668 = extern_kernels.convolution(buf667, primals_334, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (4, 1, 6, 6), (36, 36, 6, 1))
        buf669 = buf668; del buf668  # reuse
        # Topologically Sorted Source Nodes: [out_21], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_69.run(buf669, primals_335, 144, grid=grid(144), stream=stream0)
        del primals_335
        buf276 = empty_strided_cuda((), (), torch.float32)
        buf670 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [mse_loss, mul, loss], Original ATen: [aten.mse_loss, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mse_loss_mul_70.run(buf670, buf270, buf275, 1, 256, grid=grid(1), stream=stream0)
    return (buf656, buf670, buf669, primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_26, primals_27, primals_28, primals_30, primals_31, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_50, primals_51, primals_52, primals_54, primals_55, primals_56, primals_58, primals_60, primals_61, primals_62, primals_64, primals_65, primals_66, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_76, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_86, primals_88, primals_94, primals_95, primals_96, primals_98, primals_99, primals_100, primals_102, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_130, primals_131, primals_132, primals_134, primals_137, primals_139, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_149, primals_155, primals_156, primals_157, primals_159, primals_160, primals_161, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_171, primals_177, primals_178, primals_179, primals_181, primals_182, primals_183, primals_185, primals_191, primals_192, primals_193, primals_195, primals_196, primals_197, primals_199, primals_205, primals_207, primals_208, primals_209, primals_211, primals_212, primals_213, primals_215, primals_217, primals_218, primals_219, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_229, primals_230, primals_231, primals_233, primals_235, primals_236, primals_237, primals_239, primals_240, primals_241, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_251, primals_252, primals_253, primals_255, primals_256, primals_257, primals_259, primals_261, primals_262, primals_263, primals_265, primals_266, primals_267, primals_269, primals_271, primals_272, primals_273, primals_275, primals_276, primals_277, primals_279, primals_280, primals_281, primals_283, primals_284, primals_285, primals_287, primals_289, primals_290, primals_291, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_301, primals_302, primals_303, primals_305, primals_306, primals_307, primals_309, primals_310, primals_311, primals_313, primals_314, primals_315, primals_317, primals_319, primals_320, primals_321, primals_322, primals_324, primals_325, primals_326, primals_327, primals_329, primals_330, primals_331, primals_332, primals_334, buf1, buf5, buf8, buf10, buf12, buf16, buf19, buf21, buf23, buf27, buf30, buf32, buf34, buf38, buf41, buf43, buf45, buf47, buf48, buf51, buf53, buf55, buf56, buf59, buf61, buf63, buf64, buf67, buf69, buf71, buf72, buf75, buf77, buf79, buf81, buf82, buf85, buf87, buf89, buf90, buf93, buf95, buf98, buf99, buf102, buf104, buf106, buf107, buf110, buf112, buf114, buf116, buf117, buf120, buf122, buf124, buf125, buf128, buf130, buf132, buf133, buf136, buf138, buf140, buf141, buf144, buf146, buf148, buf150, buf151, buf154, buf156, buf158, buf159, buf162, buf164, buf167, reinterpret_tensor(buf168, (4, 32), (32, 1), 0), reinterpret_tensor(buf171, (4, 32), (32, 1), 0), reinterpret_tensor(buf172, (64, 512), (512, 1), 0), reinterpret_tensor(buf174, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf174, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf174, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), buf176, buf177, buf178, buf179, reinterpret_tensor(buf180, (64, 512), (512, 1), 0), buf182, buf183, buf186, buf188, buf190, buf191, buf194, buf196, buf198, reinterpret_tensor(buf199, (4, 32), (32, 1), 0), reinterpret_tensor(buf202, (4, 32), (32, 1), 0), reinterpret_tensor(buf203, (64, 512), (512, 1), 0), reinterpret_tensor(buf205, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf205, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf205, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), buf207, buf208, buf209, buf210, reinterpret_tensor(buf211, (64, 512), (512, 1), 0), buf213, buf214, buf217, buf219, buf221, buf222, buf225, buf227, buf229, reinterpret_tensor(buf230, (4, 32), (32, 1), 0), reinterpret_tensor(buf233, (4, 32), (32, 1), 0), reinterpret_tensor(buf234, (64, 512), (512, 1), 0), reinterpret_tensor(buf236, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf236, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf236, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), buf238, buf239, buf240, buf241, reinterpret_tensor(buf242, (64, 512), (512, 1), 0), buf244, buf245, buf248, buf250, buf252, buf253, buf256, buf258, buf260, buf261, buf264, buf266, buf268, buf270, reinterpret_tensor(buf274, (4, 4, 4), (16, 4, 1), 0), buf275, buf277, buf279, buf281, buf282, buf285, buf287, buf289, buf290, buf293, buf295, buf297, reinterpret_tensor(buf298, (4, 32), (32, 1), 0), reinterpret_tensor(buf301, (4, 32), (32, 1), 0), reinterpret_tensor(buf302, (64, 512), (512, 1), 0), reinterpret_tensor(buf304, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf304, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf304, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), buf306, buf307, buf308, buf309, reinterpret_tensor(buf310, (64, 512), (512, 1), 0), buf312, buf313, buf316, buf318, buf320, buf321, buf324, buf326, buf328, buf329, buf332, buf334, buf336, buf337, buf340, buf342, buf344, reinterpret_tensor(buf345, (4, 32), (32, 1), 0), reinterpret_tensor(buf348, (4, 32), (32, 1), 0), reinterpret_tensor(buf349, (64, 512), (512, 1), 0), reinterpret_tensor(buf351, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf351, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf351, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), buf353, buf354, buf355, buf356, reinterpret_tensor(buf357, (64, 512), (512, 1), 0), buf359, buf360, buf363, buf365, buf367, buf368, buf371, buf373, buf375, reinterpret_tensor(buf376, (4, 32), (32, 1), 0), reinterpret_tensor(buf379, (4, 32), (32, 1), 0), reinterpret_tensor(buf380, (64, 512), (512, 1), 0), reinterpret_tensor(buf382, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf382, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf382, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), buf384, buf385, buf386, buf387, reinterpret_tensor(buf388, (64, 512), (512, 1), 0), buf390, buf391, buf394, buf396, buf398, buf399, buf402, buf404, buf406, reinterpret_tensor(buf407, (4, 32), (32, 1), 0), reinterpret_tensor(buf410, (4, 32), (32, 1), 0), reinterpret_tensor(buf411, (64, 512), (512, 1), 0), reinterpret_tensor(buf413, (4, 1, 16, 512), (512, 32768, 2048, 1), 0), reinterpret_tensor(buf413, (4, 1, 16, 512), (512, 32768, 2048, 1), 32768), reinterpret_tensor(buf413, (4, 1, 16, 512), (512, 32768, 2048, 1), 65536), buf415, buf416, buf417, buf418, reinterpret_tensor(buf419, (64, 512), (512, 1), 0), buf421, buf422, buf424, buf425, buf428, buf430, buf432, buf433, buf436, buf438, buf441, buf442, buf445, buf447, buf449, buf450, buf453, buf455, buf457, buf458, buf461, buf463, buf465, buf466, buf469, buf471, buf473, buf474, buf476, buf477, buf480, buf482, buf484, buf485, buf488, buf490, buf492, buf493, buf496, buf498, buf500, buf501, buf504, buf506, buf508, buf509, buf512, buf514, buf516, buf517, buf520, buf522, buf524, buf525, buf527, buf528, buf531, buf533, buf535, buf536, buf539, buf541, buf544, buf545, buf548, buf550, buf552, buf553, buf556, buf558, buf560, buf561, buf564, buf566, buf568, buf569, buf572, buf574, buf576, buf577, buf579, buf583, buf586, buf588, buf590, buf594, buf597, buf599, buf601, buf605, buf608, buf610, buf612, buf616, buf619, buf621, buf623, buf627, buf630, buf632, buf634, buf638, buf641, buf643, buf645, buf649, buf652, buf654, buf656, buf658, buf659, buf661, buf662, buf664, buf665, buf667, primals_203, primals_202, primals_189, primals_188, primals_175, primals_174, primals_153, primals_152, primals_120, primals_119, primals_106, primals_105, primals_92, primals_91, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((4, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1536, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((3, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
