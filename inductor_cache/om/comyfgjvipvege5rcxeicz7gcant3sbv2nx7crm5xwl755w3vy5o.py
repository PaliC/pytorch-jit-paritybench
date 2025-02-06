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


# kernel path: inductor_cache/qx/cqxin45riywudgnttzy2cj3rcwaotyqddf5kjhetfetogourt5zr.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_2 => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_red_fused_native_group_norm_0 = async_compile.triton('triton_red_fused_native_group_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp5 = 4096.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.store(out_ptr2 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kr/ckrvkorlklatszobfom7iuax3g2vep5alryorthfe5tbuabcxbqq.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.native_group_norm, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1
#   input_3 => gt, mul_2, where
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.2), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_2), kwargs = {})
triton_poi_fused_leaky_relu_native_group_norm_1 = async_compile.triton('triton_poi_fused_leaky_relu_native_group_norm_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_native_group_norm_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_native_group_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4096.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.2
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/nu/cnuxuloaarbzyviabwxvffteptu2ytesqx45x2eujnwo7sanz7rb.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_5 => add_2, rsqrt_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_2, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
triton_per_fused_native_group_norm_2 = async_compile.triton('triton_per_fused_native_group_norm_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
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
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 1024*x0), None)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tmp6 = tl.full([1], 1024, tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 / tmp7
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 1024.0
    tmp15 = tmp13 / tmp14
    tmp16 = 1e-05
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tl.store(out_ptr2 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/am/camihm5pulsjyg3gxkmqhqekejubaj7kulzzsjoh7g6hzwo7hdlb.py
# Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.native_group_norm, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_5 => add_3, mul_4
#   input_6 => gt_1, mul_5, where_1
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %unsqueeze_11), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %unsqueeze_8), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_3, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.2), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %add_3, %mul_5), kwargs = {})
triton_poi_fused_leaky_relu_native_group_norm_3 = async_compile.triton('triton_poi_fused_leaky_relu_native_group_norm_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_native_group_norm_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_native_group_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.2
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/hq/chqvapdsfzanfcdphaj6z7szbhzrfsts2cp4oaudtbzjkwcyd37r.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_8 => add_4, rsqrt_2, var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
triton_per_fused_native_group_norm_4 = async_compile.triton('triton_per_fused_native_group_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 1, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_4(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 256
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
    tl.store(out_ptr2 + (x0), tmp18, None)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp13, None)
''', device_str='cuda')


# kernel path: inductor_cache/lr/clrl2lt6n4imlneqknwqrnccv7edes6qtthr4unc254v5kzv3pis.py
# Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.native_group_norm, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_7
#   input_9 => gt_2, mul_8, where_2
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_17), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %unsqueeze_14), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_5, 0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_5, 0.2), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_5, %mul_8), kwargs = {})
triton_poi_fused_leaky_relu_native_group_norm_5 = async_compile.triton('triton_poi_fused_leaky_relu_native_group_norm_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_native_group_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_native_group_norm_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 0.0
    tmp15 = tmp13 > tmp14
    tmp16 = 0.2
    tmp17 = tmp13 * tmp16
    tmp18 = tl.where(tmp15, tmp13, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/cs/ccsiizzt62mf552q3jjjosdoxbka5zmtd2qqx3ylhehmd4zpq6rl.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   input_11 => gt_3, mul_9, where_3
# Graph fragment:
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 0.2), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_3, %mul_9), kwargs = {})
triton_poi_fused_leaky_relu_6 = async_compile.triton('triton_poi_fused_leaky_relu_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_6(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(in_out_ptr0 + (x0), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/lz/clz2abjsldkpxof2mgyntazphnwbql35tssses7io76swg53umnb.py
# Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_13 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_4,), kwargs = {})
triton_poi_fused_relu_7 = async_compile.triton('triton_poi_fused_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_7(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nv/cnvhnuv3m5nfpn5tnixpfoeesslrkbrp4wtoylihfpoio4chxhg4.py
# Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_15 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_5,), kwargs = {})
triton_poi_fused_relu_8 = async_compile.triton('triton_poi_fused_relu_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_8(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/4k/c4kwxpekyrofkend3yyhpxlv7v3v47ybd23ocow3lgfenevh4bl2.py
# Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_19 => add_7, mul_11
#   input_20 => relu_3
# Graph fragment:
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %unsqueeze_23), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_20), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused_native_group_norm_relu_9 = async_compile.triton('triton_poi_fused_native_group_norm_relu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 64
    x1 = ((xindex // 64) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 256.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/gs/cgs42r35d2nddyyy4baih5qh6tidh7b6a4p3psqw3jx7dcbbxn42.py
# Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_25 => add_11, mul_15
#   input_26 => relu_5
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %unsqueeze_35), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_32), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused_native_group_norm_relu_10 = async_compile.triton('triton_poi_fused_native_group_norm_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1024.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/kr/ckrb426ognssxmmlduugzc3eyx5azoopedi43jp7xkp2fdegbmzg.py
# Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_31 => add_15, mul_19
#   input_32 => relu_7
# Graph fragment:
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %unsqueeze_47), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, %unsqueeze_44), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_15,), kwargs = {})
triton_poi_fused_native_group_norm_relu_11 = async_compile.triton('triton_poi_fused_native_group_norm_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 1024
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 4096.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/zf/czfdmdyngnubinb56m5zvom4ig6u44ot5ndwh7nuadc3vzlf6l7c.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_34 => add_16, rsqrt_8, var_mean_8
# Graph fragment:
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
triton_red_fused_native_group_norm_12 = async_compile.triton('triton_red_fused_native_group_norm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_12(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp5 = 4096.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xn/cxnxxlwjjf6mwjhnvg6usenyek7ubwff2evvfamoy2hdttatselg.py
# Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_36 => add_18, add_19, convert_element_type, convert_element_type_1, iota, mul_22, mul_23
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_18, torch.float32), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_19, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_23, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_13 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/7s/c7settmejruugw7z4di24epzjqvk7ryojd7lrdfumpgzje7623co.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.native_group_norm, aten.relu, aten._unsafe_index]
# Source node to ATen node mapping:
#   input_34 => add_17, mul_21
#   input_35 => relu_8
#   input_36 => _unsafe_index
# Graph fragment:
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %unsqueeze_53), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_50), kwargs = {})
#   %relu_8 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu_8, [None, None, %unsqueeze_54, %convert_element_type_1]), kwargs = {})
triton_poi_fused__unsafe_index_native_group_norm_relu_14 = async_compile.triton('triton_poi_fused__unsafe_index_native_group_norm_relu_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_native_group_norm_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_native_group_norm_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex // 4096
    x2 = ((xindex // 4096) % 64)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x5 // 4), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x5 // 4), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 32, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr1 + (tmp8 + 32*tmp4 + 1024*x5), None, eviction_policy='evict_last')
    tmp11 = tmp9 - tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x6), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/al/cal7rwyz4jsxu5i3v5xk4blebtecugua5dpom6fg5qiistcr23we.py
# Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_38 => var_mean_9
# Graph fragment:
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_native_group_norm_15 = async_compile.triton('triton_red_fused_native_group_norm_15', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_group_norm_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_group_norm_15(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 8192*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tl.store(out_ptr2 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ha/chaf7t2idd4ujih5mtyjvrldkjzwjwyz3srdszny46muemgn4gfr.py
# Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_38 => add_22, rsqrt_9, var_mean_9
# Graph fragment:
#   %var_mean_9 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_18, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_18, 1e-05), kwargs = {})
#   %rsqrt_9 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
triton_per_fused_native_group_norm_16 = async_compile.triton('triton_per_fused_native_group_norm_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4v/c4vd6idgf3vkrymp5ee4zx2qz4fzqcyzvjmq5jovucwpsdu7nqnu.py
# Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_38 => add_23, mul_27
#   input_39 => relu_9
# Graph fragment:
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %unsqueeze_60), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_57), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_23,), kwargs = {})
triton_poi_fused_native_group_norm_relu_17 = async_compile.triton('triton_poi_fused_native_group_norm_relu_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 4096
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 16384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/7b/c7bdrnbn6n3knsc5hk2jzmwewtgxermmtyl4qnqvxrj2d45dtsh6.py
# Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   input_44 => tanh
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_15,), kwargs = {})
triton_poi_fused_tanh_18 = async_compile.triton('triton_poi_fused_tanh_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_18(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = libdevice.tanh(tmp0)
    tl.store(in_out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39 = args
    args.clear()
    assert_size_stride(primals_1, (64, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_9, (256, ), (1, ))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (512, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(primals_12, (128, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(primals_13, (128, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(primals_14, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_15, (512, 256, 4, 4), (4096, 16, 4, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, 128, 4, 4), (2048, 16, 4, 1))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (64, ), (1, ))
    assert_size_stride(primals_33, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_34, (64, ), (1, ))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, 64, 5, 5), (1600, 25, 5, 1))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (64, ), (1, ))
    assert_size_stride(primals_39, (4, 64, 5, 5), (1600, 25, 5, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf2 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf5 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_0.run(buf0, buf1, buf2, buf5, 64, 4096, grid=grid(64), stream=stream0)
        buf4 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.native_group_norm, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_native_group_norm_1.run(buf6, buf0, buf1, buf2, primals_3, primals_4, 262144, grid=grid(262144), stream=stream0)
        del primals_4
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf8 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf9 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf12 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_2.run(buf7, buf8, buf9, buf12, 128, 1024, grid=grid(128), stream=stream0)
        buf11 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten.native_group_norm, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_native_group_norm_3.run(buf13, buf7, buf8, buf9, primals_6, primals_7, 131072, grid=grid(131072), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf15 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf16 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf19 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf14, buf15, buf16, buf19, 256, 256, grid=grid(256), stream=stream0)
        buf18 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        buf20 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.native_group_norm, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_native_group_norm_5.run(buf20, buf14, buf15, buf16, primals_9, primals_10, 65536, grid=grid(65536), stream=stream0)
        del primals_10
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_11, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_6.run(buf22, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 128, 1, 1), (128, 1, 1, 1))
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_7.run(buf24, 512, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf26 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_8.run(buf26, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_8.run(buf28, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_15, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf30 = buf16; del buf16  # reuse
        buf31 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf33 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf29, buf30, buf31, buf33, 256, 256, grid=grid(256), stream=stream0)
        buf34 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf29, buf30, buf31, primals_16, primals_17, buf34, 65536, grid=grid(65536), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf36 = buf31; del buf31  # reuse
        buf37 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf39 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_4.run(buf35, buf36, buf37, buf39, 256, 256, grid=grid(256), stream=stream0)
        buf40 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_9.run(buf35, buf36, buf37, primals_19, primals_20, buf40, 65536, grid=grid(65536), stream=stream0)
        del buf37
        del primals_20
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_21, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf42 = buf9; del buf9  # reuse
        buf43 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf45 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_2.run(buf41, buf42, buf43, buf45, 128, 1024, grid=grid(128), stream=stream0)
        buf46 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf41, buf42, buf43, primals_22, primals_23, buf46, 131072, grid=grid(131072), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf48 = buf43; del buf43  # reuse
        buf49 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        buf51 = empty_strided_cuda((4, 32, 1, 1), (32, 1, 128, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_2.run(buf47, buf48, buf49, buf51, 128, 1024, grid=grid(128), stream=stream0)
        buf52 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf47, buf48, buf49, primals_25, primals_26, buf52, 131072, grid=grid(131072), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_27, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf54 = buf2; del buf2  # reuse
        buf55 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf57 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_0.run(buf53, buf54, buf55, buf57, 64, 4096, grid=grid(64), stream=stream0)
        buf58 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_11.run(buf53, buf54, buf55, primals_28, primals_29, buf58, 262144, grid=grid(262144), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf60 = reinterpret_tensor(buf55, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf55  # reuse
        buf61 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf63 = reinterpret_tensor(buf61, (4, 16, 1, 1), (16, 1, 1, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_12.run(buf63, buf59, buf60, 64, 4096, grid=grid(64), stream=stream0)
        buf64 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_13.run(buf64, 64, grid=grid(64), stream=stream0)
        buf65 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.native_group_norm, aten.relu, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_native_group_norm_relu_14.run(buf64, buf59, buf60, buf63, primals_31, primals_32, buf65, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf67 = reinterpret_tensor(buf49, (4, 16, 1, 1, 2), (32, 2, 128, 128, 1), 0); del buf49  # reuse
        buf68 = empty_strided_cuda((4, 16, 1, 1, 2), (32, 2, 128, 128, 1), torch.float32)
        buf69 = empty_strided_cuda((4, 16, 1, 1, 2), (32, 2, 128, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_15.run(buf66, buf67, buf68, buf69, 128, 8192, grid=grid(128), stream=stream0)
        buf70 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf71 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf73 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_16.run(buf67, buf68, buf69, buf70, buf71, buf73, 64, 2, grid=grid(64), stream=stream0)
        buf74 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_17.run(buf66, buf70, buf71, primals_34, primals_35, buf74, 1048576, grid=grid(1048576), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_36, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf76 = buf69; del buf69  # reuse
        buf77 = buf68; del buf68  # reuse
        buf78 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_group_norm_15.run(buf75, buf76, buf77, buf78, 128, 8192, grid=grid(128), stream=stream0)
        buf79 = buf71; del buf71  # reuse
        buf80 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        buf82 = empty_strided_cuda((4, 16, 1, 1), (16, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_16.run(buf76, buf77, buf78, buf79, buf80, buf82, 64, 2, grid=grid(64), stream=stream0)
        del buf76
        del buf77
        del buf78
        buf83 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_17.run(buf75, buf79, buf80, primals_37, primals_38, buf83, 1048576, grid=grid(1048576), stream=stream0)
        del buf80
        del primals_38
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_39, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 4, 64, 64), (16384, 4096, 64, 1))
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_18.run(buf85, 65536, grid=grid(65536), stream=stream0)
    return (buf85, primals_1, primals_2, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_32, primals_33, primals_34, primals_36, primals_37, primals_39, buf0, reinterpret_tensor(buf1, (4, 16), (16, 1), 0), reinterpret_tensor(buf5, (4, 16), (16, 1), 0), buf6, buf7, reinterpret_tensor(buf8, (4, 32), (32, 1), 0), reinterpret_tensor(buf12, (4, 32), (32, 1), 0), buf13, buf14, reinterpret_tensor(buf15, (4, 64), (64, 1), 0), reinterpret_tensor(buf19, (4, 64), (64, 1), 0), buf20, buf22, buf24, buf26, buf28, buf29, reinterpret_tensor(buf30, (4, 64), (64, 1), 0), reinterpret_tensor(buf33, (4, 64), (64, 1), 0), buf34, buf35, reinterpret_tensor(buf36, (4, 64), (64, 1), 0), reinterpret_tensor(buf39, (4, 64), (64, 1), 0), buf40, buf41, reinterpret_tensor(buf42, (4, 32), (32, 1), 0), reinterpret_tensor(buf45, (4, 32), (32, 1), 0), buf46, buf47, reinterpret_tensor(buf48, (4, 32), (32, 1), 0), reinterpret_tensor(buf51, (4, 32), (32, 1), 0), buf52, buf53, reinterpret_tensor(buf54, (4, 16), (16, 1), 0), reinterpret_tensor(buf57, (4, 16), (16, 1), 0), buf58, buf59, buf60, buf63, buf64, buf65, buf66, reinterpret_tensor(buf70, (4, 16), (16, 1), 0), reinterpret_tensor(buf73, (4, 16), (16, 1), 0), buf74, buf75, reinterpret_tensor(buf79, (4, 16), (16, 1), 0), reinterpret_tensor(buf82, (4, 16), (16, 1), 0), buf83, buf85, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, 256, 4, 4), (4096, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, 128, 4, 4), (2048, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 64, 4, 4), (1024, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, 64, 5, 5), (1600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, 64, 5, 5), (1600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
