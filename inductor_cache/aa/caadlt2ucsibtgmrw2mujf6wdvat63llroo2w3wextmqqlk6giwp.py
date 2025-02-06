# AOT ID: ['13_forward']
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


# kernel path: inductor_cache/4j/c4jf3g7lz6lhicl2pcgpwqcjcf6kfscknxxofh5yofju7kulqvx5.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x => div, mul, pow_1, pow_2, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_2, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1, 2], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %pow_2), kwargs = {})
#   %mul : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %div), kwargs = {})
triton_per_fused__weight_norm_interface_0 = async_compile.triton('triton_per_fused__weight_norm_interface_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 15
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 15*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 15*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cw/ccwoofxt2yzc2k5bmsi67sxy32q57jbqynjojx3xtbyhvmik7mby.py
# Topologically Sorted Source Nodes: [conv1d, x_1], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv1d => convolution
#   x_1 => gt, mul_1, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_4, %mul, %primals_3, [1], [7], [1], False, [0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul_1), kwargs = {})
triton_poi_fused_convolution_leaky_relu_1 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/nc/cncvcivjcdxpnzaktnw4uiqbdnbp72xqlbwjcrpoiamafdod4bcf.py
# Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_2 => div_1, mul_2, pow_3, pow_4, sum_2
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_6, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [1, 2], True), kwargs = {})
#   %pow_4 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_5, %pow_4), kwargs = {})
#   %mul_2 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_6, %div_1), kwargs = {})
triton_per_fused__weight_norm_interface_2 = async_compile.triton('triton_per_fused__weight_norm_interface_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_2(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 164
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 164*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 164*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gt/cgtfowik7hjqradh4zn4m2kn5mhprwbxkhndid5zg35rfnrrkrcu.py
# Topologically Sorted Source Nodes: [conv1d_1, x_3, conv1d_8, x_18], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv1d_1 => convolution_1
#   conv1d_8 => convolution_8
#   x_18 => gt_7, mul_16, where_7
#   x_3 => gt_1, mul_3, where_1
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %mul_2, %primals_7, [4], [20], [1], False, [0], 4), kwargs = {})
#   %gt_1 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, 0.1), kwargs = {})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %convolution_1, %mul_3), kwargs = {})
#   %convolution_8 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_6, %mul_2, %primals_7, [4], [20], [1], False, [0], 4), kwargs = {})
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_8, 0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_8, 0.1), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %convolution_8, %mul_16), kwargs = {})
triton_poi_fused_convolution_leaky_relu_3 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_3', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_3(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/ld/cldmdu4yn2wndkgfb4iayn5rk6bhaf7qong5436tlht5ei2u567p.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_4 => div_2, mul_4, pow_5, pow_6, sum_3
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_9, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, [1, 2], True), kwargs = {})
#   %pow_6 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_8, %pow_6), kwargs = {})
#   %mul_4 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_9, %div_2), kwargs = {})
triton_per_fused__weight_norm_interface_4 = async_compile.triton('triton_per_fused__weight_norm_interface_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_4(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 164
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 164*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 164*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c5/cc5m3hfikshujtchf5cftqramoi62ilfkna7agz4sc7mjkcqmsog.py
# Topologically Sorted Source Nodes: [conv1d_2, x_5, conv1d_9, x_20], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv1d_2 => convolution_2
#   conv1d_9 => convolution_9
#   x_20 => gt_8, mul_18, where_8
#   x_5 => gt_2, mul_5, where_2
# Graph fragment:
#   %convolution_2 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_1, %mul_4, %primals_10, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_2, 0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_2, 0.1), kwargs = {})
#   %where_2 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %convolution_2, %mul_5), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_7, %mul_4, %primals_10, [4], [20], [1], False, [0], 16), kwargs = {})
#   %gt_8 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_9, 0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_9, 0.1), kwargs = {})
#   %where_8 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %convolution_9, %mul_18), kwargs = {})
triton_poi_fused_convolution_leaky_relu_5 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_5(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/tu/ctudycu63rx6vlt25dza4iu6dawzjlx6pjbznpb3y4qq327ystx4.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_6 => div_3, mul_6, pow_7, pow_8, sum_4
# Graph fragment:
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_12, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_7, [1, 2], True), kwargs = {})
#   %pow_8 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_11, %pow_8), kwargs = {})
#   %mul_6 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_12, %div_3), kwargs = {})
triton_per_fused__weight_norm_interface_6 = async_compile.triton('triton_per_fused__weight_norm_interface_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_6(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 164
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 164*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 164*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y5/cy5dlqobqlboc4ij3oqrsoctxq2zzylpt235kgebj35jdy53lynl.py
# Topologically Sorted Source Nodes: [conv1d_3, x_7, conv1d_10, x_22], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv1d_10 => convolution_10
#   conv1d_3 => convolution_3
#   x_22 => gt_9, mul_20, where_9
#   x_7 => gt_3, mul_7, where_3
# Graph fragment:
#   %convolution_3 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %mul_6, %primals_13, [4], [20], [1], False, [0], 64), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_3, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_3, 0.1), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %convolution_3, %mul_7), kwargs = {})
#   %convolution_10 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_8, %mul_6, %primals_13, [4], [20], [1], False, [0], 64), kwargs = {})
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_10, 0), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_10, 0.1), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %convolution_10, %mul_20), kwargs = {})
triton_poi_fused_convolution_leaky_relu_7 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_7', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_7(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x2), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x2), tmp7, None)
    tl.store(in_out_ptr1 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/il/cili5ob27bzzegz6eskgrctdxjg2qrgreuumuca6yfutjgp4w4pn.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_10 => div_5, mul_10, pow_11, pow_12, sum_6
# Graph fragment:
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_18, 2), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_11, [1, 2], True), kwargs = {})
#   %pow_12 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_6, 0.5), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_17, %pow_12), kwargs = {})
#   %mul_10 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_18, %div_5), kwargs = {})
triton_red_fused__weight_norm_interface_8 = async_compile.triton('triton_red_fused__weight_norm_interface_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 5120*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 5120*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 5120*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m7/cm7jt4n2fyitkov7skaxn7kjbcavyyqvs74xn2oih4itazg34qh2.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_12 => div_6, mul_12, pow_13, pow_14, sum_7
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_21, 2), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_13, [1, 2], True), kwargs = {})
#   %pow_14 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_7, 0.5), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_20, %pow_14), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_21, %div_6), kwargs = {})
triton_red_fused__weight_norm_interface_9 = async_compile.triton('triton_red_fused__weight_norm_interface_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_9(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp6 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tmp8 / tmp5
        tmp10 = tmp6 * tmp9
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/46/c46ae22jsyxouoxmkhvg5qpovbv43vvkdtd6rxwplh54dve6tuvz.py
# Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_13 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_5, %mul_12, %primals_22, [1], [1], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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


# kernel path: inductor_cache/hc/chcugoh7jp5cfhexmlazdbmc6vnbzu7wnv3xypdc7gwoyt62ia63.py
# Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_31 => pow_29, pow_30, sum_15
# Graph fragment:
#   %pow_29 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_25, 2), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_29, [1, 2, 3], True), kwargs = {})
#   %pow_30 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_15, 0.5), kwargs = {})
triton_poi_fused__weight_norm_interface_11 = async_compile.triton('triton_poi_fused__weight_norm_interface_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 5*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 5*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 5*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4 + 5*x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sf/csfo7vtufrypjllveh7fefq3sqwitdukplht6jiamuuirm5rqtiw.py
# Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_31 => div_14, mul_26
# Graph fragment:
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_24, %pow_30), kwargs = {})
#   %mul_26 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_25, %div_14), kwargs = {})
triton_poi_fused__weight_norm_interface_12 = async_compile.triton('triton_poi_fused__weight_norm_interface_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__weight_norm_interface_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__weight_norm_interface_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 / tmp2
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yd/cyde4gan6rgztsstnks4sicq6hg7nxoeouxbf6kygec3gl2pc3lh.py
# Topologically Sorted Source Nodes: [conv2d, x_32, conv2d_6, x_46], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d => convolution_14
#   conv2d_6 => convolution_20
#   x_32 => gt_12, mul_27, where_12
#   x_46 => gt_17, mul_38, where_17
# Graph fragment:
#   %convolution_14 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_2, %mul_26, %primals_26, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_14, 0), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_14, 0.1), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %convolution_14, %mul_27), kwargs = {})
#   %convolution_20 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_4, %mul_26, %primals_26, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_17 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_20, 0), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_20, 0.1), kwargs = {})
#   %where_17 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_17, %convolution_20, %mul_38), kwargs = {})
triton_poi_fused_convolution_leaky_relu_13 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_13(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 22) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k3/ck3tkf6ozburwdziympneswl6defztndj4mnrxmrgemtmqablxq4.py
# Topologically Sorted Source Nodes: [x_33], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_33 => div_15, mul_28, pow_31, pow_32, sum_16
# Graph fragment:
#   %pow_31 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_28, 2), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_31, [1, 2, 3], True), kwargs = {})
#   %pow_32 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_16, 0.5), kwargs = {})
#   %div_15 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_27, %pow_32), kwargs = {})
#   %mul_28 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_28, %div_15), kwargs = {})
triton_per_fused__weight_norm_interface_14 = async_compile.triton('triton_per_fused__weight_norm_interface_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_14(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 160
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 160*x0), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr0 + (r1 + 160*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/22/c22qdmzamr4qcnhktmxjaanqvindw2s2vqbwbyzduabcsk3ub4fq.py
# Topologically Sorted Source Nodes: [conv2d_1, x_34, conv2d_7, x_48], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_15
#   conv2d_7 => convolution_21
#   x_34 => gt_13, mul_29, where_13
#   x_48 => gt_18, mul_40, where_18
# Graph fragment:
#   %convolution_15 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_12, %mul_28, %primals_29, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_13 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_15, 0), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_15, 0.1), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %convolution_15, %mul_29), kwargs = {})
#   %convolution_21 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_17, %mul_28, %primals_29, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_18 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_21, 0), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_21, 0.1), kwargs = {})
#   %where_18 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_18, %convolution_21, %mul_40), kwargs = {})
triton_poi_fused_convolution_leaky_relu_15 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_15(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 8) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/ac/cackcxearhy2qyhefrmk6wn5z7brgx672r5fdfnwjxixgjjpidia.py
# Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_35 => div_16, mul_30, pow_33, pow_34, sum_17
# Graph fragment:
#   %pow_33 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_31, 2), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_33, [1, 2, 3], True), kwargs = {})
#   %pow_34 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_17, 0.5), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_30, %pow_34), kwargs = {})
#   %mul_30 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_31, %div_16), kwargs = {})
triton_per_fused__weight_norm_interface_16 = async_compile.triton('triton_per_fused__weight_norm_interface_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__weight_norm_interface_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__weight_norm_interface_16(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 640
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 640*x0), rmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = triton_helpers.promote_to_tensor(tl.sum(tmp4, 0))
    tmp6 = libdevice.sqrt(tmp5)
    tmp8 = tmp7 / tmp6
    tmp9 = tmp0 * tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr0 + (r1 + 640*x0), tmp9, rmask)
''', device_str='cuda')


# kernel path: inductor_cache/lr/clrdn3gponl4bt2vmgp3erio4kt4rckqrg4imufruumhhx67mx5r.py
# Topologically Sorted Source Nodes: [conv2d_2, x_36, conv2d_8, x_50], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_2 => convolution_16
#   conv2d_8 => convolution_22
#   x_36 => gt_14, mul_31, where_14
#   x_50 => gt_19, mul_42, where_19
# Graph fragment:
#   %convolution_16 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_13, %mul_30, %primals_32, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_16, 0), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_16, 0.1), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %convolution_16, %mul_31), kwargs = {})
#   %convolution_22 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_18, %mul_30, %primals_32, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_19 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_22, 0), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_22, 0.1), kwargs = {})
#   %where_19 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_19, %convolution_22, %mul_42), kwargs = {})
triton_poi_fused_convolution_leaky_relu_17 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_17(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/ac/cacfmr2qev5vwlkr3uio2xzu5vite7jijgq67npxv45gd4r5emtv.py
# Topologically Sorted Source Nodes: [x_37], Original ATen: [aten._weight_norm_interface]
# Source node to ATen node mapping:
#   x_37 => div_17, mul_32, pow_35, pow_36, sum_18
# Graph fragment:
#   %pow_35 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_34, 2), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_35, [1, 2, 3], True), kwargs = {})
#   %pow_36 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_18, 0.5), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_33, %pow_36), kwargs = {})
#   %mul_32 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_34, %div_17), kwargs = {})
triton_red_fused__weight_norm_interface_18 = async_compile.triton('triton_red_fused__weight_norm_interface_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__weight_norm_interface_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__weight_norm_interface_18(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 2560*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0 * tmp0
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = libdevice.sqrt(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tmp7 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + 2560*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp7 / tmp5
        tmp9 = tmp6 * tmp8
        tl.store(out_ptr0 + (r1 + 2560*x0), tmp9, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a5/ca5zg7ip5fsbighahaqlqqkltpduxnnk6biyai4ue7azzcde3d3k.py
# Topologically Sorted Source Nodes: [conv2d_3, x_38, conv2d_9, x_52], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_3 => convolution_17
#   conv2d_9 => convolution_23
#   x_38 => gt_15, mul_33, where_15
#   x_52 => gt_20, mul_44, where_20
# Graph fragment:
#   %convolution_17 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_14, %mul_32, %primals_35, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_15 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_17, 0), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_17, 0.1), kwargs = {})
#   %where_15 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_15, %convolution_17, %mul_33), kwargs = {})
#   %convolution_23 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_19, %mul_32, %primals_35, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_20 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_23, 0), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_23, 0.1), kwargs = {})
#   %where_20 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_20, %convolution_23, %mul_44), kwargs = {})
triton_poi_fused_convolution_leaky_relu_19 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_19(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 2) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/5w/c5wbk7erlc4myolwixlncfr3ldluh345z6ccidrkfdv3pwcjmqi3.py
# Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_42 => convolution_19
# Graph fragment:
#   %convolution_19 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_16, %mul_36, %primals_41, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_20 = async_compile.triton('triton_poi_fused_convolution_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
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


# kernel path: inductor_cache/dr/cdrflzgnvqwccexvqrbakm3flbo2trtrfajbhmjultiiwf2plpto.py
# Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_58 => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad1d_21 = async_compile.triton('triton_poi_fused_reflection_pad1d_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = xindex // 66
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7q/c7qmvmestte3qrmwo5a2j25s6jq2lm34qw3zqrol3irj2dm7apw4.py
# Topologically Sorted Source Nodes: [conv2d_12, x_61, conv2d_18, x_76], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_12 => convolution_26
#   conv2d_18 => convolution_32
#   x_61 => gt_22, mul_49, where_22
#   x_76 => gt_27, mul_60, where_27
# Graph fragment:
#   %convolution_26 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_6, %mul_48, %primals_44, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_22 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_26, 0), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_26, 0.1), kwargs = {})
#   %where_22 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_22, %convolution_26, %mul_49), kwargs = {})
#   %convolution_32 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_8, %mul_48, %primals_44, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_27 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_32, 0), kwargs = {})
#   %mul_60 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_32, 0.1), kwargs = {})
#   %where_27 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_27, %convolution_32, %mul_60), kwargs = {})
triton_poi_fused_convolution_leaky_relu_22 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_22', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_22(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 24) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rk/crkcqtg4yxqa5yhg3pq4xawlk6ymcv3vaw3gumu24xbt6uqcqtvt.py
# Topologically Sorted Source Nodes: [conv2d_13, x_63, conv2d_19, x_78], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_13 => convolution_27
#   conv2d_19 => convolution_33
#   x_63 => gt_23, mul_51, where_23
#   x_78 => gt_28, mul_62, where_28
# Graph fragment:
#   %convolution_27 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_22, %mul_50, %primals_47, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_23 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_27, 0), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_27, 0.1), kwargs = {})
#   %where_23 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_23, %convolution_27, %mul_51), kwargs = {})
#   %convolution_33 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_27, %mul_50, %primals_47, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_28 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_33, 0), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_33, 0.1), kwargs = {})
#   %where_28 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_28, %convolution_33, %mul_62), kwargs = {})
triton_poi_fused_convolution_leaky_relu_23 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_23(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 9) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lg/clguxhnop6oacvs6dba3ll3nhw2kgcl2uow62jxxxeueoeixeuyu.py
# Topologically Sorted Source Nodes: [conv2d_14, x_65, conv2d_20, x_80], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_14 => convolution_28
#   conv2d_20 => convolution_34
#   x_65 => gt_24, mul_53, where_24
#   x_80 => gt_29, mul_64, where_29
# Graph fragment:
#   %convolution_28 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_23, %mul_52, %primals_50, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_24 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_28, 0), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_28, 0.1), kwargs = {})
#   %where_24 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_24, %convolution_28, %mul_53), kwargs = {})
#   %convolution_34 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_28, %mul_52, %primals_50, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_29 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_34, 0), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_34, 0.1), kwargs = {})
#   %where_29 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_29, %convolution_34, %mul_64), kwargs = {})
triton_poi_fused_convolution_leaky_relu_24 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_24', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_24(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 3) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crz2rmud7vfxgcb6xzhaautuazuahwvxidfeoyhjdmbfsjumdpgu.py
# Topologically Sorted Source Nodes: [conv2d_15, x_67, conv2d_21, x_82], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_15 => convolution_29
#   conv2d_21 => convolution_35
#   x_67 => gt_25, mul_55, where_25
#   x_82 => gt_30, mul_66, where_30
# Graph fragment:
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_24, %mul_54, %primals_53, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_25 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_29, 0), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_29, 0.1), kwargs = {})
#   %where_25 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_25, %convolution_29, %mul_55), kwargs = {})
#   %convolution_35 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_29, %mul_54, %primals_53, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_30 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_35, 0), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_35, 0.1), kwargs = {})
#   %where_30 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_30, %convolution_35, %mul_66), kwargs = {})
triton_poi_fused_convolution_leaky_relu_25 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_25(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 3) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/5x/c5xmzrqu74xe7ua3wutnctsc7jmhpwv34wgs5sgzgpbwi5ukp5c2.py
# Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_71 => convolution_31
# Graph fragment:
#   %convolution_31 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_26, %mul_58, %primals_59, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_26 = async_compile.triton('triton_poi_fused_convolution_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_26(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12
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


# kernel path: inductor_cache/z2/cz2hq64dgnvh6tuarvgkemwtbp2lrnpwel462nchjdqedqf4nz7r.py
# Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_88 => _unsafe_index_2
# Graph fragment:
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_5]), kwargs = {})
triton_poi_fused_reflection_pad1d_27 = async_compile.triton('triton_poi_fused_reflection_pad1d_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 260
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 65)
    x1 = xindex // 65
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rg/crgvztaxdodd4gtmize2ta2ldwi6e5zl6pkpqvdn4af57mymau5c.py
# Topologically Sorted Source Nodes: [conv2d_24, x_91, conv2d_30, x_106], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_24 => convolution_38
#   conv2d_30 => convolution_44
#   x_106 => gt_37, mul_82, where_37
#   x_91 => gt_32, mul_71, where_32
# Graph fragment:
#   %convolution_38 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_10, %mul_70, %primals_62, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_32 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_38, 0), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_38, 0.1), kwargs = {})
#   %where_32 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_32, %convolution_38, %mul_71), kwargs = {})
#   %convolution_44 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_12, %mul_70, %primals_62, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_37 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_44, 0), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_44, 0.1), kwargs = {})
#   %where_37 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_37, %convolution_44, %mul_82), kwargs = {})
triton_poi_fused_convolution_leaky_relu_28 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_28', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_28(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 25) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gk/cgkqh6ffgwguj6aneup4rbgmk6tyz5advo3tqjhlam2dw2igumd4.py
# Topologically Sorted Source Nodes: [conv2d_25, x_93, conv2d_31, x_108], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_25 => convolution_39
#   conv2d_31 => convolution_45
#   x_108 => gt_38, mul_84, where_38
#   x_93 => gt_33, mul_73, where_33
# Graph fragment:
#   %convolution_39 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_32, %mul_72, %primals_65, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_33 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_39, 0), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_39, 0.1), kwargs = {})
#   %where_33 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_33, %convolution_39, %mul_73), kwargs = {})
#   %convolution_45 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_37, %mul_72, %primals_65, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_38 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_45, 0), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_45, 0.1), kwargs = {})
#   %where_38 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_38, %convolution_45, %mul_84), kwargs = {})
triton_poi_fused_convolution_leaky_relu_29 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_29', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_29(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 10) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/46/c4657myedb7zmumb7gok6ntjbvdgva3y7tp72ww77mqn2waw27dc.py
# Topologically Sorted Source Nodes: [conv2d_26, x_95, conv2d_32, x_110], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_26 => convolution_40
#   conv2d_32 => convolution_46
#   x_110 => gt_39, mul_86, where_39
#   x_95 => gt_34, mul_75, where_34
# Graph fragment:
#   %convolution_40 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_33, %mul_74, %primals_68, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_34 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_40, 0), kwargs = {})
#   %mul_75 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_40, 0.1), kwargs = {})
#   %where_34 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_34, %convolution_40, %mul_75), kwargs = {})
#   %convolution_46 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_38, %mul_74, %primals_68, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_39 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_46, 0), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_46, 0.1), kwargs = {})
#   %where_39 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_39, %convolution_46, %mul_86), kwargs = {})
triton_poi_fused_convolution_leaky_relu_30 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_30', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_30(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 5) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v6/cv636fvivjwvycq2amsvpi5jduepjqwvcbfnnsuccxmmklepwok7.py
# Topologically Sorted Source Nodes: [conv2d_27, x_97, conv2d_33, x_112], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_27 => convolution_41
#   conv2d_33 => convolution_47
#   x_112 => gt_40, mul_88, where_40
#   x_97 => gt_35, mul_77, where_35
# Graph fragment:
#   %convolution_41 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_34, %mul_76, %primals_71, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_35 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_41, 0), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_41, 0.1), kwargs = {})
#   %where_35 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_35, %convolution_41, %mul_77), kwargs = {})
#   %convolution_47 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_39, %mul_76, %primals_71, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_40 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_47, 0), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_47, 0.1), kwargs = {})
#   %where_40 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_40, %convolution_47, %mul_88), kwargs = {})
triton_poi_fused_convolution_leaky_relu_31 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_31(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 5) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/zb/czbm7kvxbouhtzyou7byw2akbjrhldbdab6oe62i6umuaqpe5b7w.py
# Topologically Sorted Source Nodes: [x_101], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_101 => convolution_43
# Graph fragment:
#   %convolution_43 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_36, %mul_80, %primals_77, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_32 = async_compile.triton('triton_poi_fused_convolution_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_32(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
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


# kernel path: inductor_cache/rh/crhszjkgqy2empzbvvcn5n353gyvm65qatfvk7jkygsegsawtg7b.py
# Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_118 => _unsafe_index_4
# Graph fragment:
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_9]), kwargs = {})
triton_poi_fused_reflection_pad1d_33 = async_compile.triton('triton_poi_fused_reflection_pad1d_33', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 70)
    x1 = xindex // 70
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3q/c3qtf63ebcqp7fop6jqcbrsglg2dcadhvg4blmckxijk7eek4qt7.py
# Topologically Sorted Source Nodes: [conv2d_36, x_121, conv2d_42, x_136], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_36 => convolution_50
#   conv2d_42 => convolution_56
#   x_121 => gt_42, mul_93, where_42
#   x_136 => gt_47, mul_104, where_47
# Graph fragment:
#   %convolution_50 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_14, %mul_92, %primals_80, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_42 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_50, 0), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_50, 0.1), kwargs = {})
#   %where_42 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_42, %convolution_50, %mul_93), kwargs = {})
#   %convolution_56 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_16, %mul_92, %primals_80, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_47 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_56, 0), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_56, 0.1), kwargs = {})
#   %where_47 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_47, %convolution_56, %mul_104), kwargs = {})
triton_poi_fused_convolution_leaky_relu_34 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_34', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_34(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 28) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bb/cbb3w5t5owdi3imkro2ivzlffk57seqlwp4nnge27ynjz757v6bu.py
# Topologically Sorted Source Nodes: [conv2d_37, x_123, conv2d_43, x_138], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_37 => convolution_51
#   conv2d_43 => convolution_57
#   x_123 => gt_43, mul_95, where_43
#   x_138 => gt_48, mul_106, where_48
# Graph fragment:
#   %convolution_51 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_42, %mul_94, %primals_83, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_43 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_51, 0), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_51, 0.1), kwargs = {})
#   %where_43 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_43, %convolution_51, %mul_95), kwargs = {})
#   %convolution_57 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_47, %mul_94, %primals_83, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_48 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_57, 0), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_57, 0.1), kwargs = {})
#   %where_48 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_48, %convolution_57, %mul_106), kwargs = {})
triton_poi_fused_convolution_leaky_relu_35 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_35', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_35(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 14) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chws26ncpifljtjjrsabn5zhsmqcpj6ov7kwhz5zzvt6tu4zodbq.py
# Topologically Sorted Source Nodes: [conv2d_38, x_125, conv2d_44, x_140], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_38 => convolution_52
#   conv2d_44 => convolution_58
#   x_125 => gt_44, mul_97, where_44
#   x_140 => gt_49, mul_108, where_49
# Graph fragment:
#   %convolution_52 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_43, %mul_96, %primals_86, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_44 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_52, 0), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_52, 0.1), kwargs = {})
#   %where_44 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_44, %convolution_52, %mul_97), kwargs = {})
#   %convolution_58 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_48, %mul_96, %primals_86, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_49 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_58, 0), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_58, 0.1), kwargs = {})
#   %where_49 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_49, %convolution_58, %mul_108), kwargs = {})
triton_poi_fused_convolution_leaky_relu_36 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_36', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_36(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 7) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bh/cbhxqxe4ns6ayqjrfv4st2s42y7hovh6tdp4lzfehiqgy2camn72.py
# Topologically Sorted Source Nodes: [conv2d_39, x_127, conv2d_45, x_142], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_39 => convolution_53
#   conv2d_45 => convolution_59
#   x_127 => gt_45, mul_99, where_45
#   x_142 => gt_50, mul_110, where_50
# Graph fragment:
#   %convolution_53 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_44, %mul_98, %primals_89, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_45 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_53, 0), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_53, 0.1), kwargs = {})
#   %where_45 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_45, %convolution_53, %mul_99), kwargs = {})
#   %convolution_59 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_49, %mul_98, %primals_89, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_50 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_59, 0), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_59, 0.1), kwargs = {})
#   %where_50 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_50, %convolution_59, %mul_110), kwargs = {})
triton_poi_fused_convolution_leaky_relu_37 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_37', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_37(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 7) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/sq/csqchrn3nxiupekzfka56pz5xvpiwvch2po4fwc2qzjrug4ivws2.py
# Topologically Sorted Source Nodes: [x_131], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_131 => convolution_55
# Graph fragment:
#   %convolution_55 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_46, %mul_102, %primals_95, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_38 = async_compile.triton('triton_poi_fused_convolution_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_38(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28
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


# kernel path: inductor_cache/e3/ce32l3iqugvn5uyozxynuespwo3v5jx6jv4j7ljpowf2cgdobfxv.py
# Topologically Sorted Source Nodes: [conv2d_49, x_153, conv2d_55, x_168], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_49 => convolution_63
#   conv2d_55 => convolution_69
#   x_153 => gt_53, mul_117, where_53
#   x_168 => gt_58, mul_128, where_58
# Graph fragment:
#   %convolution_63 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_52, %mul_116, %primals_101, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_53 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_63, 0), kwargs = {})
#   %mul_117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_63, 0.1), kwargs = {})
#   %where_53 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_53, %convolution_63, %mul_117), kwargs = {})
#   %convolution_69 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_57, %mul_116, %primals_101, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_58 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_69, 0), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_69, 0.1), kwargs = {})
#   %where_58 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_58, %convolution_69, %mul_128), kwargs = {})
triton_poi_fused_convolution_leaky_relu_39 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_39', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_39(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 11) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qq/cqqn4arquydhpyza2jeqaub4e4jr4ihhzfl6jr4xpvphfeg4zwup.py
# Topologically Sorted Source Nodes: [conv2d_50, x_155, conv2d_56, x_170], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_50 => convolution_64
#   conv2d_56 => convolution_70
#   x_155 => gt_54, mul_119, where_54
#   x_170 => gt_59, mul_130, where_59
# Graph fragment:
#   %convolution_64 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_53, %mul_118, %primals_104, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_54 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_64, 0), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_64, 0.1), kwargs = {})
#   %where_54 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_54, %convolution_64, %mul_119), kwargs = {})
#   %convolution_70 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_58, %mul_118, %primals_104, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_59 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_70, 0), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_70, 0.1), kwargs = {})
#   %where_59 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_59, %convolution_70, %mul_130), kwargs = {})
triton_poi_fused_convolution_leaky_relu_40 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_40', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_40(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 11) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/in/cinknk5aoojkbfg65kxcc2fyhkzr2savid4i6updl7dall7pzwp4.py
# Topologically Sorted Source Nodes: [conv2d_51, x_157, conv2d_57, x_172], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_51 => convolution_65
#   conv2d_57 => convolution_71
#   x_157 => gt_55, mul_121, where_55
#   x_172 => gt_60, mul_132, where_60
# Graph fragment:
#   %convolution_65 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_54, %mul_120, %primals_107, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_55 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_65, 0), kwargs = {})
#   %mul_121 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_65, 0.1), kwargs = {})
#   %where_55 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_55, %convolution_65, %mul_121), kwargs = {})
#   %convolution_71 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_59, %mul_120, %primals_107, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_60 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_71, 0), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_71, 0.1), kwargs = {})
#   %where_60 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_60, %convolution_71, %mul_132), kwargs = {})
triton_poi_fused_convolution_leaky_relu_41 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_41', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_41(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 45056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 11) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/w7/cw7zghc5ysltp73cvewkcc3svcewxw25qqvlfwayzcwekkgoreny.py
# Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_161 => convolution_67
# Graph fragment:
#   %convolution_67 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_56, %mul_124, %primals_113, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_42 = async_compile.triton('triton_poi_fused_convolution_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_42(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 44
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


# kernel path: inductor_cache/bi/cbilo4smelgi34bugnhcumnpvnkm4zzjeqyou5du5hqnspwh7ghc.py
# Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_178 => _unsafe_index_8
# Graph fragment:
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_17]), kwargs = {})
triton_poi_fused_reflection_pad1d_43 = async_compile.triton('triton_poi_fused_reflection_pad1d_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_43(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 68)
    x1 = xindex // 68
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gj/cgjr5wea2zqw5qqrqmq6zurli7g4xngrcumk6c3cuhj63r6l4mop.py
# Topologically Sorted Source Nodes: [conv2d_60, x_181], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_60 => convolution_74
#   x_181 => gt_62, mul_137, where_62
# Graph fragment:
#   %convolution_74 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_22, %mul_136, %primals_116, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_62 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_74, 0), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_74, 0.1), kwargs = {})
#   %where_62 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_62, %convolution_74, %mul_137), kwargs = {})
triton_poi_fused_convolution_leaky_relu_44 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_44(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 34) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wa/cwas2w27xgem2p5bmik2sukeyefo5vylueji2m3x24x6wzmzzmjk.py
# Topologically Sorted Source Nodes: [conv2d_61, x_183, conv2d_67, x_198], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_61 => convolution_75
#   conv2d_67 => convolution_81
#   x_183 => gt_63, mul_139, where_63
#   x_198 => gt_68, mul_150, where_68
# Graph fragment:
#   %convolution_75 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_62, %mul_138, %primals_119, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_63 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_75, 0), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_75, 0.1), kwargs = {})
#   %where_63 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_63, %convolution_75, %mul_139), kwargs = {})
#   %convolution_81 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_67, %mul_138, %primals_119, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_68 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_81, 0), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_81, 0.1), kwargs = {})
#   %where_68 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_68, %convolution_81, %mul_150), kwargs = {})
triton_poi_fused_convolution_leaky_relu_45 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_45', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_45', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_45(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 17) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ac/cacr2g75mkgzcr3of3ptcz6zchdikixv6uzlch577edegwoxtnz6.py
# Topologically Sorted Source Nodes: [conv2d_62, x_185, conv2d_68, x_200], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_62 => convolution_76
#   conv2d_68 => convolution_82
#   x_185 => gt_64, mul_141, where_64
#   x_200 => gt_69, mul_152, where_69
# Graph fragment:
#   %convolution_76 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_63, %mul_140, %primals_122, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_64 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_76, 0), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_76, 0.1), kwargs = {})
#   %where_64 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_64, %convolution_76, %mul_141), kwargs = {})
#   %convolution_82 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_68, %mul_140, %primals_122, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_69 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_82, 0), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_82, 0.1), kwargs = {})
#   %where_69 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_69, %convolution_82, %mul_152), kwargs = {})
triton_poi_fused_convolution_leaky_relu_46 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_46', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_46(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 17) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c6/cc6iyabwkjebuogmoryvasoh23qfxlacrzzbr4kn6jdj2dyj37vm.py
# Topologically Sorted Source Nodes: [conv2d_63, x_187, conv2d_69, x_202], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_63 => convolution_77
#   conv2d_69 => convolution_83
#   x_187 => gt_65, mul_143, where_65
#   x_202 => gt_70, mul_154, where_70
# Graph fragment:
#   %convolution_77 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_64, %mul_142, %primals_125, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_65 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_77, 0), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_77, 0.1), kwargs = {})
#   %where_65 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_65, %convolution_77, %mul_143), kwargs = {})
#   %convolution_83 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_69, %mul_142, %primals_125, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_70 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_83, 0), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_83, 0.1), kwargs = {})
#   %where_70 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_70, %convolution_83, %mul_154), kwargs = {})
triton_poi_fused_convolution_leaky_relu_47 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_47', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_47', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_47(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 69632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 17) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/gv/cgvl7zbfdakxzyaqj52hjvdw2tq673xg5ayppeskpt2yzr2ihzm6.py
# Topologically Sorted Source Nodes: [x_191], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_191 => convolution_79
# Graph fragment:
#   %convolution_79 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_66, %mul_146, %primals_131, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_48 = async_compile.triton('triton_poi_fused_convolution_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_48(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 68
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


# kernel path: inductor_cache/q3/cq3tswwk4vkb33pntbsfmnmi2tjmxozdknyw7dz2zwew4mbx7me4.py
# Topologically Sorted Source Nodes: [x_208], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_208 => _unsafe_index_10
# Graph fragment:
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_21]), kwargs = {})
triton_poi_fused_reflection_pad1d_49 = async_compile.triton('triton_poi_fused_reflection_pad1d_49', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 276
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 69)
    x1 = xindex // 69
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gi/cgi7my5bxpur4g5g5ask6ixnd53f3oka56acnynh7y42whmq3pvg.py
# Topologically Sorted Source Nodes: [conv2d_72, x_211, conv2d_78, x_226], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_72 => convolution_86
#   conv2d_78 => convolution_92
#   x_211 => gt_72, mul_159, where_72
#   x_226 => gt_77, mul_170, where_77
# Graph fragment:
#   %convolution_86 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_26, %mul_158, %primals_134, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_72 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_86, 0), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_86, 0.1), kwargs = {})
#   %where_72 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_72, %convolution_86, %mul_159), kwargs = {})
#   %convolution_92 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_28, %mul_158, %primals_134, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_77 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_92, 0), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_92, 0.1), kwargs = {})
#   %where_77 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_77, %convolution_92, %mul_170), kwargs = {})
triton_poi_fused_convolution_leaky_relu_50 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_50', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_50(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 23) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxlewt4o6sdyvqfu6iyyw6f536nh7f2thblykxyyhxgogatcoyn.py
# Topologically Sorted Source Nodes: [conv2d_73, x_213, conv2d_79, x_228], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_73 => convolution_87
#   conv2d_79 => convolution_93
#   x_213 => gt_73, mul_161, where_73
#   x_228 => gt_78, mul_172, where_78
# Graph fragment:
#   %convolution_87 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_72, %mul_160, %primals_137, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_73 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_87, 0), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_87, 0.1), kwargs = {})
#   %where_73 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_73, %convolution_87, %mul_161), kwargs = {})
#   %convolution_93 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_77, %mul_160, %primals_137, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_78 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_93, 0), kwargs = {})
#   %mul_172 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_93, 0.1), kwargs = {})
#   %where_78 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_78, %convolution_93, %mul_172), kwargs = {})
triton_poi_fused_convolution_leaky_relu_51 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_51', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_51', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_51(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 23) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ey/ceyrq7eslzgtk5i673puu4isgsknn7xnwhfmtscxbgx4o7fj5tzc.py
# Topologically Sorted Source Nodes: [conv2d_74, x_215, conv2d_80, x_230], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_74 => convolution_88
#   conv2d_80 => convolution_94
#   x_215 => gt_74, mul_163, where_74
#   x_230 => gt_79, mul_174, where_79
# Graph fragment:
#   %convolution_88 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_73, %mul_162, %primals_140, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_74 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_88, 0), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_88, 0.1), kwargs = {})
#   %where_74 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_74, %convolution_88, %mul_163), kwargs = {})
#   %convolution_94 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_78, %mul_162, %primals_140, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_79 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_94, 0), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_94, 0.1), kwargs = {})
#   %where_79 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_79, %convolution_94, %mul_174), kwargs = {})
triton_poi_fused_convolution_leaky_relu_52 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_52', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_52', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_52(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 47104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 23) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
    tl.store(in_out_ptr1 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zi/czioa3ylwa4tcfevmxm2w6yktieph6rjmhv2ptzek47smlmkxupx.py
# Topologically Sorted Source Nodes: [conv2d_75, x_217, conv2d_81, x_232], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_75 => convolution_89
#   conv2d_81 => convolution_95
#   x_217 => gt_75, mul_165, where_75
#   x_232 => gt_80, mul_176, where_80
# Graph fragment:
#   %convolution_89 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_74, %mul_164, %primals_143, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_75 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_89, 0), kwargs = {})
#   %mul_165 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_89, 0.1), kwargs = {})
#   %where_75 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_75, %convolution_89, %mul_165), kwargs = {})
#   %convolution_95 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_79, %mul_164, %primals_143, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_80 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_95, 0), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_95, 0.1), kwargs = {})
#   %where_80 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_80, %convolution_95, %mul_176), kwargs = {})
triton_poi_fused_convolution_leaky_relu_53 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_53', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_53(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 94208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 23) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = tmp9 > tmp3
    tmp11 = tmp9 * tmp5
    tmp12 = tl.where(tmp10, tmp9, tmp11)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
    tl.store(in_out_ptr1 + (x3), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/al/caltb6xkbk6hubadd7l74pugncb62uwlmvf5bpu6civ3punu5i3x.py
# Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_221 => convolution_91
# Graph fragment:
#   %convolution_91 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_76, %mul_168, %primals_149, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_54 = async_compile.triton('triton_poi_fused_convolution_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_54(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 92
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


# kernel path: inductor_cache/ys/cysqov6c6ljaxieekbefuieazy5l5lvuud6plaxmr5f5ammoxnzi.py
# Topologically Sorted Source Nodes: [x_238], Original ATen: [aten.reflection_pad1d]
# Source node to ATen node mapping:
#   x_238 => _unsafe_index_12
# Graph fragment:
#   %_unsafe_index_12 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_4, [None, None, %sub_25]), kwargs = {})
triton_poi_fused_reflection_pad1d_55 = async_compile.triton('triton_poi_fused_reflection_pad1d_55', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad1d_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad1d_55(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 74)
    x1 = xindex // 74
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (63 + ((-1)*tl_math.abs((-63) + x0)) + 64*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uv/cuvtcbmkmmhzfzrrmljk7wf5smwz2dj4t7nh74eiwcsfakjhs6et.py
# Topologically Sorted Source Nodes: [conv2d_84, x_241], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_84 => convolution_98
#   x_241 => gt_82, mul_181, where_82
# Graph fragment:
#   %convolution_98 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%view_30, %mul_180, %primals_152, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_82 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_98, 0), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_98, 0.1), kwargs = {})
#   %where_82 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_82, %convolution_98, %mul_181), kwargs = {})
triton_poi_fused_convolution_leaky_relu_56 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_56(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 37) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sd/csd7kwbsemdnui53cfccr5i6eogvhq4hqob4h6gxeufknykqulhp.py
# Topologically Sorted Source Nodes: [conv2d_85, x_243], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_85 => convolution_99
#   x_243 => gt_83, mul_183, where_83
# Graph fragment:
#   %convolution_99 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_82, %mul_182, %primals_155, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_83 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_99, 0), kwargs = {})
#   %mul_183 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_99, 0.1), kwargs = {})
#   %where_83 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_83, %convolution_99, %mul_183), kwargs = {})
triton_poi_fused_convolution_leaky_relu_57 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_57', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_57(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 37) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kl/cklcru54zczovhhu7xbo3l6lcriidvy7qijx7hwtlfzpfju4sewz.py
# Topologically Sorted Source Nodes: [conv2d_86, x_245], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_86 => convolution_100
#   x_245 => gt_84, mul_185, where_84
# Graph fragment:
#   %convolution_100 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_83, %mul_184, %primals_158, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_84 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_100, 0), kwargs = {})
#   %mul_185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_100, 0.1), kwargs = {})
#   %where_84 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_84, %convolution_100, %mul_185), kwargs = {})
triton_poi_fused_convolution_leaky_relu_58 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_58(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 37) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dg/cdgfo2uuyuxpsma473pjgsg2axxtgltzcd3362yadkup5u6ydapp.py
# Topologically Sorted Source Nodes: [conv2d_87, x_247], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   conv2d_87 => convolution_101
#   x_247 => gt_85, mul_187, where_85
# Graph fragment:
#   %convolution_101 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%where_84, %mul_186, %primals_161, [3, 1], [2, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt_85 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_101, 0), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_101, 0.1), kwargs = {})
#   %where_85 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_85, %convolution_101, %mul_187), kwargs = {})
triton_poi_fused_convolution_leaky_relu_59 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_59(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 151552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 37) % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.1
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/67/c67olwrjfssnosh2ya5ik6w4r6cm72j74jcof6ljgd6ak3mdycj3.py
# Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_251 => convolution_103
# Graph fragment:
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_86, %mul_190, %primals_167, [1, 1], [1, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_60 = async_compile.triton('triton_poi_fused_convolution_60', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_60', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_60(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 148
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167 = args
    args.clear()
    assert_size_stride(primals_1, (16, 1, 1), (1, 1, 1))
    assert_size_stride(primals_2, (16, 1, 15), (15, 15, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_5, (64, 1, 1), (1, 1, 1))
    assert_size_stride(primals_6, (64, 4, 41), (164, 41, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (256, 1, 1), (1, 1, 1))
    assert_size_stride(primals_9, (256, 4, 41), (164, 41, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_12, (1024, 4, 41), (164, 41, 1))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_15, (1024, 4, 41), (164, 41, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    assert_size_stride(primals_17, (1024, 1, 1), (1, 1, 1))
    assert_size_stride(primals_18, (1024, 1024, 5), (5120, 5, 1))
    assert_size_stride(primals_19, (1024, ), (1, ))
    assert_size_stride(primals_20, (1, 1, 1), (1, 1, 1))
    assert_size_stride(primals_21, (1, 1024, 3), (3072, 3, 1))
    assert_size_stride(primals_22, (1, ), (1, ))
    assert_size_stride(primals_23, (4, 1, 64), (64, 64, 1))
    assert_size_stride(primals_24, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_25, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_27, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_28, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_31, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_34, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_37, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_38, (1024, ), (1, ))
    assert_size_stride(primals_39, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_40, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_41, (1, ), (1, ))
    assert_size_stride(primals_42, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_43, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_46, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_49, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_52, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_53, (1024, ), (1, ))
    assert_size_stride(primals_54, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_55, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_56, (1024, ), (1, ))
    assert_size_stride(primals_57, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_58, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_59, (1, ), (1, ))
    assert_size_stride(primals_60, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_61, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_62, (32, ), (1, ))
    assert_size_stride(primals_63, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_64, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_67, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_70, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_71, (1024, ), (1, ))
    assert_size_stride(primals_72, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_73, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_74, (1024, ), (1, ))
    assert_size_stride(primals_75, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_76, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_77, (1, ), (1, ))
    assert_size_stride(primals_78, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_79, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_80, (32, ), (1, ))
    assert_size_stride(primals_81, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_82, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_85, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_88, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_89, (1024, ), (1, ))
    assert_size_stride(primals_90, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_91, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_94, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_95, (1, ), (1, ))
    assert_size_stride(primals_96, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_97, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_98, (32, ), (1, ))
    assert_size_stride(primals_99, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_100, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_103, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_106, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_107, (1024, ), (1, ))
    assert_size_stride(primals_108, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_109, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_112, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_113, (1, ), (1, ))
    assert_size_stride(primals_114, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_115, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_116, (32, ), (1, ))
    assert_size_stride(primals_117, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_118, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_121, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_124, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_125, (1024, ), (1, ))
    assert_size_stride(primals_126, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_127, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_130, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_131, (1, ), (1, ))
    assert_size_stride(primals_132, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_133, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_134, (32, ), (1, ))
    assert_size_stride(primals_135, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_136, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_139, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_140, (512, ), (1, ))
    assert_size_stride(primals_141, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_142, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_143, (1024, ), (1, ))
    assert_size_stride(primals_144, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_145, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_148, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_149, (1, ), (1, ))
    assert_size_stride(primals_150, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_151, (32, 1, 5, 1), (5, 5, 1, 1))
    assert_size_stride(primals_152, (32, ), (1, ))
    assert_size_stride(primals_153, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_154, (128, 32, 5, 1), (160, 5, 1, 1))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_157, (512, 128, 5, 1), (640, 5, 1, 1))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_159, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_160, (1024, 512, 5, 1), (2560, 5, 1, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_163, (1024, 1024, 5, 1), (5120, 5, 1, 1))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_165, (1, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_166, (1, 1024, 3, 1), (3072, 3, 1, 1))
    assert_size_stride(primals_167, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 1, 1), (1, 16, 16), torch.float32)
        buf1 = reinterpret_tensor(buf0, (16, 1, 1), (1, 1, 1), 0); del buf0  # reuse
        buf2 = empty_strided_cuda((16, 1, 15), (15, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_0.run(buf1, primals_2, primals_1, buf2, 16, 15, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(primals_4, buf2, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (4, 16, 64), (1024, 64, 1))
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [conv1d, x_1], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf4, primals_3, 4096, grid=grid(4096), stream=stream0)
        buf5 = empty_strided_cuda((64, 1, 1), (1, 64, 64), torch.float32)
        buf6 = reinterpret_tensor(buf5, (64, 1, 1), (1, 1, 1), 0); del buf5  # reuse
        buf7 = empty_strided_cuda((64, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_2.run(buf6, primals_6, primals_5, buf7, 64, 164, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_1], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf4, buf7, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf8, (4, 64, 16), (1024, 16, 1))
        # Topologically Sorted Source Nodes: [conv1d_7], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(primals_23, buf2, stride=(1,), padding=(7,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf35, (4, 16, 64), (1024, 64, 1))
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [conv1d_7, x_16], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_1.run(buf36, primals_3, 4096, grid=grid(4096), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [conv1d_8], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, buf7, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=4, bias=None)
        assert_size_stride(buf37, (4, 64, 16), (1024, 16, 1))
        buf9 = buf8; del buf8  # reuse
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [conv1d_1, x_3, conv1d_8, x_18], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_3.run(buf9, buf38, primals_7, 4096, grid=grid(4096), stream=stream0)
        del primals_7
        buf10 = empty_strided_cuda((256, 1, 1), (1, 256, 256), torch.float32)
        buf11 = reinterpret_tensor(buf10, (256, 1, 1), (1, 1, 1), 0); del buf10  # reuse
        buf12 = empty_strided_cuda((256, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_4.run(buf11, primals_9, primals_8, buf12, 256, 164, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_2], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf9, buf12, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf13, (4, 256, 4), (1024, 4, 1))
        # Topologically Sorted Source Nodes: [conv1d_9], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, buf12, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=16, bias=None)
        assert_size_stride(buf39, (4, 256, 4), (1024, 4, 1))
        buf14 = buf13; del buf13  # reuse
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [conv1d_2, x_5, conv1d_9, x_20], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_5.run(buf14, buf40, primals_10, 4096, grid=grid(4096), stream=stream0)
        del primals_10
        buf15 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf16 = reinterpret_tensor(buf15, (1024, 1, 1), (1, 1, 1), 0); del buf15  # reuse
        buf17 = empty_strided_cuda((1024, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_6.run(buf16, primals_12, primals_11, buf17, 1024, 164, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_3], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf14, buf17, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=64, bias=None)
        assert_size_stride(buf18, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [conv1d_10], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, buf17, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=64, bias=None)
        assert_size_stride(buf41, (4, 1024, 1), (1024, 1, 1))
        buf19 = buf18; del buf18  # reuse
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [conv1d_3, x_7, conv1d_10, x_22], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_7.run(buf19, buf42, primals_13, 4096, grid=grid(4096), stream=stream0)
        del primals_13
        buf20 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf21 = reinterpret_tensor(buf20, (1024, 1, 1), (1, 1, 1), 0); del buf20  # reuse
        buf22 = empty_strided_cuda((1024, 4, 41), (164, 41, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_6.run(buf21, primals_15, primals_14, buf22, 1024, 164, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_4], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf19, buf22, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=256, bias=None)
        assert_size_stride(buf23, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [conv1d_11], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf22, stride=(4,), padding=(20,), dilation=(1,), transposed=False, output_padding=(0,), groups=256, bias=None)
        assert_size_stride(buf43, (4, 1024, 1), (1024, 1, 1))
        buf24 = buf23; del buf23  # reuse
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [conv1d_4, x_9, conv1d_11, x_24], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_7.run(buf24, buf44, primals_16, 4096, grid=grid(4096), stream=stream0)
        del primals_16
        buf25 = empty_strided_cuda((1024, 1, 1), (1, 1024, 1024), torch.float32)
        buf26 = reinterpret_tensor(buf25, (1024, 1, 1), (1, 1, 1), 0); del buf25  # reuse
        buf27 = empty_strided_cuda((1024, 1024, 5), (5120, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf26, primals_18, primals_17, buf27, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_5], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf24, buf27, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf28, (4, 1024, 1), (1024, 1, 1))
        # Topologically Sorted Source Nodes: [conv1d_12], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, buf27, stride=(1,), padding=(2,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf45, (4, 1024, 1), (1024, 1, 1))
        buf29 = buf28; del buf28  # reuse
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [conv1d_5, x_11, conv1d_12, x_26], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_7.run(buf29, buf46, primals_19, 4096, grid=grid(4096), stream=stream0)
        del primals_19
        buf30 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.float32)
        buf31 = buf30; del buf30  # reuse
        buf32 = empty_strided_cuda((1, 1024, 3), (3072, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf31, primals_21, primals_20, buf32, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf29, buf32, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf33, (4, 1, 1), (1, 1, 1))
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(buf34, primals_22, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, buf32, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf47, (4, 1, 1), (1, 1, 1))
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(buf48, primals_22, 4, grid=grid(4), stream=stream0)
        del primals_22
        buf49 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_25, buf49, 32, grid=grid(32), stream=stream0)
        buf50 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_25, primals_24, buf49, buf50, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(reinterpret_tensor(primals_4, (4, 1, 32, 2), (64, 64, 2, 1), 0), buf50, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 32, 11, 2), (704, 22, 2, 1))
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(reinterpret_tensor(primals_23, (4, 1, 32, 2), (64, 64, 2, 1), 0), buf50, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 32, 11, 2), (704, 22, 2, 1))
        buf52 = buf51; del buf51  # reuse
        buf79 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [conv2d, x_32, conv2d_6, x_46], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_13.run(buf52, buf79, primals_26, 2816, grid=grid(2816), stream=stream0)
        del primals_26
        buf53 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf54 = reinterpret_tensor(buf53, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf53  # reuse
        buf55 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf54, primals_28, primals_27, buf55, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf52, buf55, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 128, 4, 2), (1024, 8, 2, 1))
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, buf55, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 128, 4, 2), (1024, 8, 2, 1))
        buf57 = buf56; del buf56  # reuse
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1, x_34, conv2d_7, x_48], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_15.run(buf57, buf81, primals_29, 4096, grid=grid(4096), stream=stream0)
        del primals_29
        buf58 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf59 = reinterpret_tensor(buf58, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf58  # reuse
        buf60 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf59, primals_31, primals_30, buf60, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf57, buf60, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 512, 2, 2), (2048, 4, 2, 1))
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, buf60, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 512, 2, 2), (2048, 4, 2, 1))
        buf62 = buf61; del buf61  # reuse
        buf83 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [conv2d_2, x_36, conv2d_8, x_50], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_17.run(buf62, buf83, primals_32, 8192, grid=grid(8192), stream=stream0)
        del primals_32
        buf63 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf64 = reinterpret_tensor(buf63, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf63  # reuse
        buf65 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf64, primals_34, primals_33, buf65, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf62, buf65, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 1024, 1, 2), (2048, 2, 2, 1))
        # Topologically Sorted Source Nodes: [conv2d_9], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, buf65, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 1024, 1, 2), (2048, 2, 2, 1))
        buf67 = buf66; del buf66  # reuse
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [conv2d_3, x_38, conv2d_9, x_52], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_19.run(buf67, buf85, primals_35, 8192, grid=grid(8192), stream=stream0)
        del primals_35
        buf68 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf69 = reinterpret_tensor(buf68, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf68  # reuse
        buf70 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf69, primals_37, primals_36, buf70, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf67, buf70, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 1024, 1, 2), (2048, 2, 2, 1))
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, buf70, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 1024, 1, 2), (2048, 2, 2, 1))
        buf72 = buf71; del buf71  # reuse
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [conv2d_4, x_40, conv2d_10, x_54], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_19.run(buf72, buf87, primals_38, 8192, grid=grid(8192), stream=stream0)
        del primals_38
        buf73 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf74 = buf73; del buf73  # reuse
        buf75 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf74, primals_40, primals_39, buf75, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf72, buf75, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 1, 1, 2), (2, 2, 2, 1))
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf77, primals_41, 8, grid=grid(8), stream=stream0)
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, buf75, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 1, 1, 2), (2, 2, 2, 1))
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_20.run(buf89, primals_41, 8, grid=grid(8), stream=stream0)
        del primals_41
        buf90 = empty_strided_cuda((4, 1, 66), (66, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_21.run(primals_4, buf90, 264, grid=grid(264), stream=stream0)
        buf91 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_43, buf91, 32, grid=grid(32), stream=stream0)
        buf92 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_43, primals_42, buf91, buf92, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(reinterpret_tensor(buf90, (4, 1, 22, 3), (66, 66, 3, 1), 0), buf92, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 32, 8, 3), (768, 24, 3, 1))
        buf120 = empty_strided_cuda((4, 1, 66), (66, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_73], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_21.run(primals_23, buf120, 264, grid=grid(264), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_18], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(reinterpret_tensor(buf120, (4, 1, 22, 3), (66, 66, 3, 1), 0), buf92, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 32, 8, 3), (768, 24, 3, 1))
        buf94 = buf93; del buf93  # reuse
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [conv2d_12, x_61, conv2d_18, x_76], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_22.run(buf94, buf122, primals_44, 3072, grid=grid(3072), stream=stream0)
        del primals_44
        buf95 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf96 = reinterpret_tensor(buf95, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf95  # reuse
        buf97 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf96, primals_46, primals_45, buf97, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_13], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf94, buf97, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 128, 3, 3), (1152, 9, 3, 1))
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, buf97, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 128, 3, 3), (1152, 9, 3, 1))
        buf99 = buf98; del buf98  # reuse
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [conv2d_13, x_63, conv2d_19, x_78], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_23.run(buf99, buf124, primals_47, 4608, grid=grid(4608), stream=stream0)
        del primals_47
        buf100 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf101 = reinterpret_tensor(buf100, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf100  # reuse
        buf102 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf101, primals_49, primals_48, buf102, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf99, buf102, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 512, 1, 3), (1536, 3, 3, 1))
        # Topologically Sorted Source Nodes: [conv2d_20], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, buf102, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 512, 1, 3), (1536, 3, 3, 1))
        buf104 = buf103; del buf103  # reuse
        buf126 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [conv2d_14, x_65, conv2d_20, x_80], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_24.run(buf104, buf126, primals_50, 6144, grid=grid(6144), stream=stream0)
        del primals_50
        buf105 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf106 = reinterpret_tensor(buf105, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf105  # reuse
        buf107 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf106, primals_52, primals_51, buf107, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf104, buf107, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 1024, 1, 3), (3072, 3, 3, 1))
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, buf107, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 1024, 1, 3), (3072, 3, 3, 1))
        buf109 = buf108; del buf108  # reuse
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [conv2d_15, x_67, conv2d_21, x_82], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_25.run(buf109, buf128, primals_53, 12288, grid=grid(12288), stream=stream0)
        del primals_53
        buf110 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf111 = reinterpret_tensor(buf110, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf110  # reuse
        buf112 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf111, primals_55, primals_54, buf112, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_16], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf109, buf112, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 1024, 1, 3), (3072, 3, 3, 1))
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, buf112, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 1024, 1, 3), (3072, 3, 3, 1))
        buf114 = buf113; del buf113  # reuse
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [conv2d_16, x_69, conv2d_22, x_84], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_25.run(buf114, buf130, primals_56, 12288, grid=grid(12288), stream=stream0)
        del primals_56
        buf115 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf116 = buf115; del buf115  # reuse
        buf117 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf116, primals_58, primals_57, buf117, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf114, buf117, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 1, 1, 3), (3, 3, 3, 1))
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_26.run(buf119, primals_59, 12, grid=grid(12), stream=stream0)
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, buf117, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 1, 1, 3), (3, 3, 3, 1))
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_26.run(buf132, primals_59, 12, grid=grid(12), stream=stream0)
        del primals_59
        buf133 = empty_strided_cuda((4, 1, 65), (65, 65, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_27.run(primals_4, buf133, 260, grid=grid(260), stream=stream0)
        buf134 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_61, buf134, 32, grid=grid(32), stream=stream0)
        buf135 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_61, primals_60, buf134, buf135, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_24], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(reinterpret_tensor(buf133, (4, 1, 13, 5), (65, 0, 5, 1), 0), buf135, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 32, 5, 5), (800, 25, 5, 1))
        buf163 = empty_strided_cuda((4, 1, 65), (65, 65, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_103], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_27.run(primals_23, buf163, 260, grid=grid(260), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(reinterpret_tensor(buf163, (4, 1, 13, 5), (65, 0, 5, 1), 0), buf135, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 32, 5, 5), (800, 25, 5, 1))
        buf137 = buf136; del buf136  # reuse
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [conv2d_24, x_91, conv2d_30, x_106], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_28.run(buf137, buf165, primals_62, 3200, grid=grid(3200), stream=stream0)
        del primals_62
        buf138 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf139 = reinterpret_tensor(buf138, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf138  # reuse
        buf140 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf139, primals_64, primals_63, buf140, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf137, buf140, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 128, 2, 5), (1280, 10, 5, 1))
        # Topologically Sorted Source Nodes: [conv2d_31], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, buf140, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 128, 2, 5), (1280, 10, 5, 1))
        buf142 = buf141; del buf141  # reuse
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [conv2d_25, x_93, conv2d_31, x_108], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_29.run(buf142, buf167, primals_65, 5120, grid=grid(5120), stream=stream0)
        del primals_65
        buf143 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf144 = reinterpret_tensor(buf143, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf143  # reuse
        buf145 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf144, primals_67, primals_66, buf145, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf142, buf145, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 512, 1, 5), (2560, 5, 5, 1))
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, buf145, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 512, 1, 5), (2560, 5, 5, 1))
        buf147 = buf146; del buf146  # reuse
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [conv2d_26, x_95, conv2d_32, x_110], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_30.run(buf147, buf169, primals_68, 10240, grid=grid(10240), stream=stream0)
        del primals_68
        buf148 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf149 = reinterpret_tensor(buf148, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf148  # reuse
        buf150 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf149, primals_70, primals_69, buf150, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_27], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf147, buf150, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 1024, 1, 5), (5120, 5, 5, 1))
        # Topologically Sorted Source Nodes: [conv2d_33], Original ATen: [aten.convolution]
        buf170 = extern_kernels.convolution(buf169, buf150, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (4, 1024, 1, 5), (5120, 5, 5, 1))
        buf152 = buf151; del buf151  # reuse
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [conv2d_27, x_97, conv2d_33, x_112], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_31.run(buf152, buf171, primals_71, 20480, grid=grid(20480), stream=stream0)
        del primals_71
        buf153 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf154 = reinterpret_tensor(buf153, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf153  # reuse
        buf155 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf154, primals_73, primals_72, buf155, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf152, buf155, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 1024, 1, 5), (5120, 5, 5, 1))
        # Topologically Sorted Source Nodes: [conv2d_34], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, buf155, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 1024, 1, 5), (5120, 5, 5, 1))
        buf157 = buf156; del buf156  # reuse
        buf173 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [conv2d_28, x_99, conv2d_34, x_114], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_31.run(buf157, buf173, primals_74, 20480, grid=grid(20480), stream=stream0)
        del primals_74
        buf158 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf159 = buf158; del buf158  # reuse
        buf160 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf159, primals_76, primals_75, buf160, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf157, buf160, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 1, 1, 5), (5, 5, 5, 1))
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_32.run(buf162, primals_77, 20, grid=grid(20), stream=stream0)
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, buf160, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 1, 1, 5), (5, 5, 5, 1))
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_32.run(buf175, primals_77, 20, grid=grid(20), stream=stream0)
        del primals_77
        buf176 = empty_strided_cuda((4, 1, 70), (70, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_33.run(primals_4, buf176, 280, grid=grid(280), stream=stream0)
        buf177 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_79, buf177, 32, grid=grid(32), stream=stream0)
        buf178 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_79, primals_78, buf177, buf178, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_36], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(reinterpret_tensor(buf176, (4, 1, 10, 7), (70, 0, 7, 1), 0), buf178, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 32, 4, 7), (896, 28, 7, 1))
        buf206 = empty_strided_cuda((4, 1, 70), (70, 70, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_133], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_33.run(primals_23, buf206, 280, grid=grid(280), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_42], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(reinterpret_tensor(buf206, (4, 1, 10, 7), (70, 0, 7, 1), 0), buf178, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 32, 4, 7), (896, 28, 7, 1))
        buf180 = buf179; del buf179  # reuse
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [conv2d_36, x_121, conv2d_42, x_136], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_34.run(buf180, buf208, primals_80, 3584, grid=grid(3584), stream=stream0)
        del primals_80
        buf181 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf182 = reinterpret_tensor(buf181, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf181  # reuse
        buf183 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf182, primals_82, primals_81, buf183, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_37], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf180, buf183, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 128, 2, 7), (1792, 14, 7, 1))
        # Topologically Sorted Source Nodes: [conv2d_43], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, buf183, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 128, 2, 7), (1792, 14, 7, 1))
        buf185 = buf184; del buf184  # reuse
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [conv2d_37, x_123, conv2d_43, x_138], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_35.run(buf185, buf210, primals_83, 7168, grid=grid(7168), stream=stream0)
        del primals_83
        buf186 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf187 = reinterpret_tensor(buf186, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf186  # reuse
        buf188 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf187, primals_85, primals_84, buf188, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_38], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf185, buf188, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 512, 1, 7), (3584, 7, 7, 1))
        # Topologically Sorted Source Nodes: [conv2d_44], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, buf188, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 512, 1, 7), (3584, 7, 7, 1))
        buf190 = buf189; del buf189  # reuse
        buf212 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [conv2d_38, x_125, conv2d_44, x_140], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_36.run(buf190, buf212, primals_86, 14336, grid=grid(14336), stream=stream0)
        del primals_86
        buf191 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf192 = reinterpret_tensor(buf191, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf191  # reuse
        buf193 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf192, primals_88, primals_87, buf193, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_39], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf190, buf193, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 1024, 1, 7), (7168, 7, 7, 1))
        # Topologically Sorted Source Nodes: [conv2d_45], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, buf193, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 1024, 1, 7), (7168, 7, 7, 1))
        buf195 = buf194; del buf194  # reuse
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [conv2d_39, x_127, conv2d_45, x_142], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_37.run(buf195, buf214, primals_89, 28672, grid=grid(28672), stream=stream0)
        del primals_89
        buf196 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf197 = reinterpret_tensor(buf196, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf196  # reuse
        buf198 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf197, primals_91, primals_90, buf198, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_40], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf195, buf198, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 1024, 1, 7), (7168, 7, 7, 1))
        # Topologically Sorted Source Nodes: [conv2d_46], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, buf198, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 1024, 1, 7), (7168, 7, 7, 1))
        buf200 = buf199; del buf199  # reuse
        buf216 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [conv2d_40, x_129, conv2d_46, x_144], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_37.run(buf200, buf216, primals_92, 28672, grid=grid(28672), stream=stream0)
        del primals_92
        buf201 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf202 = buf201; del buf201  # reuse
        buf203 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf202, primals_94, primals_93, buf203, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_131], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf200, buf203, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 1, 1, 7), (7, 7, 7, 1))
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [x_131], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf205, primals_95, 28, grid=grid(28), stream=stream0)
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, buf203, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 1, 1, 7), (7, 7, 7, 1))
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_38.run(buf218, primals_95, 28, grid=grid(28), stream=stream0)
        del primals_95
        buf219 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_97, buf219, 32, grid=grid(32), stream=stream0)
        buf220 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_97, primals_96, buf219, buf220, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_48], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(reinterpret_tensor(buf90, (4, 1, 6, 11), (66, 66, 11, 1), 0), buf220, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 32, 2, 11), (704, 22, 11, 1))
        # Topologically Sorted Source Nodes: [conv2d_54], Original ATen: [aten.convolution]
        buf248 = extern_kernels.convolution(reinterpret_tensor(buf120, (4, 1, 6, 11), (66, 66, 11, 1), 0), buf220, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf248, (4, 32, 2, 11), (704, 22, 11, 1))
        buf222 = buf221; del buf221  # reuse
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [conv2d_48, x_151, conv2d_54, x_166], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_13.run(buf222, buf249, primals_98, 2816, grid=grid(2816), stream=stream0)
        del primals_98
        buf223 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf224 = reinterpret_tensor(buf223, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf223  # reuse
        buf225 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf224, primals_100, primals_99, buf225, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_49], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf222, buf225, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 128, 1, 11), (1408, 11, 11, 1))
        # Topologically Sorted Source Nodes: [conv2d_55], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, buf225, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 128, 1, 11), (1408, 11, 11, 1))
        buf227 = buf226; del buf226  # reuse
        buf251 = buf250; del buf250  # reuse
        # Topologically Sorted Source Nodes: [conv2d_49, x_153, conv2d_55, x_168], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_39.run(buf227, buf251, primals_101, 5632, grid=grid(5632), stream=stream0)
        del primals_101
        buf228 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf229 = reinterpret_tensor(buf228, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf228  # reuse
        buf230 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf229, primals_103, primals_102, buf230, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_50], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf227, buf230, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 512, 1, 11), (5632, 11, 11, 1))
        # Topologically Sorted Source Nodes: [conv2d_56], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, buf230, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 512, 1, 11), (5632, 11, 11, 1))
        buf232 = buf231; del buf231  # reuse
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [conv2d_50, x_155, conv2d_56, x_170], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_40.run(buf232, buf253, primals_104, 22528, grid=grid(22528), stream=stream0)
        del primals_104
        buf233 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf234 = reinterpret_tensor(buf233, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf233  # reuse
        buf235 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf234, primals_106, primals_105, buf235, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_51], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf232, buf235, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 1024, 1, 11), (11264, 11, 11, 1))
        # Topologically Sorted Source Nodes: [conv2d_57], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, buf235, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 1024, 1, 11), (11264, 11, 11, 1))
        buf237 = buf236; del buf236  # reuse
        buf255 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [conv2d_51, x_157, conv2d_57, x_172], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_41.run(buf237, buf255, primals_107, 45056, grid=grid(45056), stream=stream0)
        del primals_107
        buf238 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf239 = reinterpret_tensor(buf238, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf238  # reuse
        buf240 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf239, primals_109, primals_108, buf240, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_52], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf237, buf240, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 1024, 1, 11), (11264, 11, 11, 1))
        # Topologically Sorted Source Nodes: [conv2d_58], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, buf240, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 1024, 1, 11), (11264, 11, 11, 1))
        buf242 = buf241; del buf241  # reuse
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [conv2d_52, x_159, conv2d_58, x_174], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_41.run(buf242, buf257, primals_110, 45056, grid=grid(45056), stream=stream0)
        del primals_110
        buf243 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf244 = buf243; del buf243  # reuse
        buf245 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf244, primals_112, primals_111, buf245, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf242, buf245, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf246, (4, 1, 1, 11), (11, 11, 11, 1))
        buf247 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf247, primals_113, 44, grid=grid(44), stream=stream0)
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, buf245, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 1, 1, 11), (11, 11, 11, 1))
        buf259 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_42.run(buf259, primals_113, 44, grid=grid(44), stream=stream0)
        del primals_113
        buf260 = empty_strided_cuda((4, 1, 68), (68, 68, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_43.run(primals_4, buf260, 272, grid=grid(272), stream=stream0)
        buf261 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_115, buf261, 32, grid=grid(32), stream=stream0)
        buf262 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_115, primals_114, buf261, buf262, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_60], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(reinterpret_tensor(buf260, (4, 1, 4, 17), (68, 0, 17, 1), 0), buf262, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 32, 2, 17), (1088, 34, 17, 1))
        buf264 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [conv2d_60, x_181], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_44.run(buf264, primals_116, 4352, grid=grid(4352), stream=stream0)
        buf265 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf266 = reinterpret_tensor(buf265, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf265  # reuse
        buf267 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf266, primals_118, primals_117, buf267, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_61], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf264, buf267, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 128, 1, 17), (2176, 17, 17, 1))
        buf290 = empty_strided_cuda((4, 1, 68), (68, 68, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_193], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_43.run(primals_23, buf290, 272, grid=grid(272), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_66], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(reinterpret_tensor(buf290, (4, 1, 4, 17), (68, 0, 17, 1), 0), buf262, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 32, 2, 17), (1088, 34, 17, 1))
        buf292 = buf291; del buf291  # reuse
        # Topologically Sorted Source Nodes: [conv2d_66, x_196], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_44.run(buf292, primals_116, 4352, grid=grid(4352), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [conv2d_67], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, buf267, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 128, 1, 17), (2176, 17, 17, 1))
        buf269 = buf268; del buf268  # reuse
        buf294 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [conv2d_61, x_183, conv2d_67, x_198], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_45.run(buf269, buf294, primals_119, 8704, grid=grid(8704), stream=stream0)
        del primals_119
        buf270 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf271 = reinterpret_tensor(buf270, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf270  # reuse
        buf272 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf271, primals_121, primals_120, buf272, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_62], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf269, buf272, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 512, 1, 17), (8704, 17, 17, 1))
        # Topologically Sorted Source Nodes: [conv2d_68], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, buf272, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 512, 1, 17), (8704, 17, 17, 1))
        buf274 = buf273; del buf273  # reuse
        buf296 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [conv2d_62, x_185, conv2d_68, x_200], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_46.run(buf274, buf296, primals_122, 34816, grid=grid(34816), stream=stream0)
        del primals_122
        buf275 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf276 = reinterpret_tensor(buf275, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf275  # reuse
        buf277 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf276, primals_124, primals_123, buf277, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_63], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf274, buf277, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 1024, 1, 17), (17408, 17, 17, 1))
        # Topologically Sorted Source Nodes: [conv2d_69], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, buf277, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 1024, 1, 17), (17408, 17, 17, 1))
        buf279 = buf278; del buf278  # reuse
        buf298 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [conv2d_63, x_187, conv2d_69, x_202], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_47.run(buf279, buf298, primals_125, 69632, grid=grid(69632), stream=stream0)
        del primals_125
        buf280 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf281 = reinterpret_tensor(buf280, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf280  # reuse
        buf282 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf281, primals_127, primals_126, buf282, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_64], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf279, buf282, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 1024, 1, 17), (17408, 17, 17, 1))
        # Topologically Sorted Source Nodes: [conv2d_70], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, buf282, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 1024, 1, 17), (17408, 17, 17, 1))
        buf284 = buf283; del buf283  # reuse
        buf300 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [conv2d_64, x_189, conv2d_70, x_204], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_47.run(buf284, buf300, primals_128, 69632, grid=grid(69632), stream=stream0)
        del primals_128
        buf285 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf286 = buf285; del buf285  # reuse
        buf287 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_190], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf286, primals_130, primals_129, buf287, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_191], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf284, buf287, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 1, 1, 17), (17, 17, 17, 1))
        buf289 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [x_191], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_48.run(buf289, primals_131, 68, grid=grid(68), stream=stream0)
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, buf287, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 1, 1, 17), (17, 17, 17, 1))
        buf302 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_48.run(buf302, primals_131, 68, grid=grid(68), stream=stream0)
        del primals_131
        buf303 = empty_strided_cuda((4, 1, 69), (69, 69, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_208], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_49.run(primals_4, buf303, 276, grid=grid(276), stream=stream0)
        buf304 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_133, buf304, 32, grid=grid(32), stream=stream0)
        buf305 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_210], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_133, primals_132, buf304, buf305, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_72], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(reinterpret_tensor(buf303, (4, 1, 3, 23), (69, 0, 23, 1), 0), buf305, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 32, 1, 23), (736, 23, 23, 1))
        buf333 = empty_strided_cuda((4, 1, 69), (69, 69, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_223], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_49.run(primals_23, buf333, 276, grid=grid(276), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_78], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(reinterpret_tensor(buf333, (4, 1, 3, 23), (69, 0, 23, 1), 0), buf305, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 32, 1, 23), (736, 23, 23, 1))
        buf307 = buf306; del buf306  # reuse
        buf335 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [conv2d_72, x_211, conv2d_78, x_226], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_50.run(buf307, buf335, primals_134, 2944, grid=grid(2944), stream=stream0)
        del primals_134
        buf308 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf309 = reinterpret_tensor(buf308, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf308  # reuse
        buf310 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_212], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf309, primals_136, primals_135, buf310, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_73], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf307, buf310, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (4, 128, 1, 23), (2944, 23, 23, 1))
        # Topologically Sorted Source Nodes: [conv2d_79], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, buf310, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 128, 1, 23), (2944, 23, 23, 1))
        buf312 = buf311; del buf311  # reuse
        buf337 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [conv2d_73, x_213, conv2d_79, x_228], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_51.run(buf312, buf337, primals_137, 11776, grid=grid(11776), stream=stream0)
        del primals_137
        buf313 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf314 = reinterpret_tensor(buf313, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf313  # reuse
        buf315 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_214], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf314, primals_139, primals_138, buf315, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_74], Original ATen: [aten.convolution]
        buf316 = extern_kernels.convolution(buf312, buf315, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 512, 1, 23), (11776, 23, 23, 1))
        # Topologically Sorted Source Nodes: [conv2d_80], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf337, buf315, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf338, (4, 512, 1, 23), (11776, 23, 23, 1))
        buf317 = buf316; del buf316  # reuse
        buf339 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [conv2d_74, x_215, conv2d_80, x_230], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_52.run(buf317, buf339, primals_140, 47104, grid=grid(47104), stream=stream0)
        del primals_140
        buf318 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf319 = reinterpret_tensor(buf318, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf318  # reuse
        buf320 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_216], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf319, primals_142, primals_141, buf320, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_75], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf317, buf320, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (4, 1024, 1, 23), (23552, 23, 23, 1))
        # Topologically Sorted Source Nodes: [conv2d_81], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, buf320, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (4, 1024, 1, 23), (23552, 23, 23, 1))
        buf322 = buf321; del buf321  # reuse
        buf341 = buf340; del buf340  # reuse
        # Topologically Sorted Source Nodes: [conv2d_75, x_217, conv2d_81, x_232], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_53.run(buf322, buf341, primals_143, 94208, grid=grid(94208), stream=stream0)
        del primals_143
        buf323 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf324 = reinterpret_tensor(buf323, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf323  # reuse
        buf325 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_218], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf324, primals_145, primals_144, buf325, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_76], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf322, buf325, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 1024, 1, 23), (23552, 23, 23, 1))
        # Topologically Sorted Source Nodes: [conv2d_82], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, buf325, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 1024, 1, 23), (23552, 23, 23, 1))
        buf327 = buf326; del buf326  # reuse
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [conv2d_76, x_219, conv2d_82, x_234], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_53.run(buf327, buf343, primals_146, 94208, grid=grid(94208), stream=stream0)
        del primals_146
        buf328 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf329 = buf328; del buf328  # reuse
        buf330 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_220], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf329, primals_148, primals_147, buf330, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf327, buf330, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 1, 1, 23), (23, 23, 23, 1))
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [x_221], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_54.run(buf332, primals_149, 92, grid=grid(92), stream=stream0)
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, buf330, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 1, 1, 23), (23, 23, 23, 1))
        buf345 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [x_236], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_54.run(buf345, primals_149, 92, grid=grid(92), stream=stream0)
        del primals_149
        buf346 = empty_strided_cuda((4, 1, 74), (74, 74, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_238], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_55.run(primals_4, buf346, 296, grid=grid(296), stream=stream0)
        buf347 = empty_strided_cuda((32, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_11.run(primals_151, buf347, 32, grid=grid(32), stream=stream0)
        buf348 = empty_strided_cuda((32, 1, 5, 1), (5, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_240], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_poi_fused__weight_norm_interface_12.run(primals_151, primals_150, buf347, buf348, 160, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_84], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(reinterpret_tensor(buf346, (4, 1, 2, 37), (74, 0, 37, 1), 0), buf348, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (4, 32, 1, 37), (1184, 37, 37, 1))
        buf350 = buf349; del buf349  # reuse
        # Topologically Sorted Source Nodes: [conv2d_84, x_241], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_56.run(buf350, primals_152, 4736, grid=grid(4736), stream=stream0)
        buf351 = empty_strided_cuda((128, 1, 1, 1), (1, 128, 128, 128), torch.float32)
        buf352 = reinterpret_tensor(buf351, (128, 1, 1, 1), (1, 1, 1, 1), 0); del buf351  # reuse
        buf353 = empty_strided_cuda((128, 32, 5, 1), (160, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_14.run(buf352, primals_154, primals_153, buf353, 128, 160, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_85], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf350, buf353, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 128, 1, 37), (4736, 37, 37, 1))
        buf355 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [conv2d_85, x_243], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_57.run(buf355, primals_155, 18944, grid=grid(18944), stream=stream0)
        buf356 = empty_strided_cuda((512, 1, 1, 1), (1, 512, 512, 512), torch.float32)
        buf357 = reinterpret_tensor(buf356, (512, 1, 1, 1), (1, 1, 1, 1), 0); del buf356  # reuse
        buf358 = empty_strided_cuda((512, 128, 5, 1), (640, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_244], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_per_fused__weight_norm_interface_16.run(buf357, primals_157, primals_156, buf358, 512, 640, grid=grid(512), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_86], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf355, buf358, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 512, 1, 37), (18944, 37, 37, 1))
        buf360 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [conv2d_86, x_245], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_58.run(buf360, primals_158, 75776, grid=grid(75776), stream=stream0)
        buf361 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf362 = reinterpret_tensor(buf361, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf361  # reuse
        buf363 = empty_strided_cuda((1024, 512, 5, 1), (2560, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_246], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_18.run(buf362, primals_160, primals_159, buf363, 1024, 2560, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_87], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf360, buf363, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (4, 1024, 1, 37), (37888, 37, 37, 1))
        buf365 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [conv2d_87, x_247], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf365, primals_161, 151552, grid=grid(151552), stream=stream0)
        buf366 = empty_strided_cuda((1024, 1, 1, 1), (1, 1024, 1024, 1024), torch.float32)
        buf367 = reinterpret_tensor(buf366, (1024, 1, 1, 1), (1, 1, 1, 1), 0); del buf366  # reuse
        buf368 = empty_strided_cuda((1024, 1024, 5, 1), (5120, 5, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_8.run(buf367, primals_163, primals_162, buf368, 1024, 5120, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_88], Original ATen: [aten.convolution]
        buf369 = extern_kernels.convolution(buf365, buf368, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (4, 1024, 1, 37), (37888, 37, 37, 1))
        buf370 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [conv2d_88, x_249], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf370, primals_164, 151552, grid=grid(151552), stream=stream0)
        buf371 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf372 = buf371; del buf371  # reuse
        buf373 = empty_strided_cuda((1, 1024, 3, 1), (3072, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_250], Original ATen: [aten._weight_norm_interface]
        stream0 = get_raw_stream(0)
        triton_red_fused__weight_norm_interface_9.run(buf372, primals_166, primals_165, buf373, 1, 3072, grid=grid(1), stream=stream0)
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.convolution]
        buf374 = extern_kernels.convolution(buf370, buf373, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf374, (4, 1, 1, 37), (37, 37, 37, 1))
        buf375 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_60.run(buf375, primals_167, 148, grid=grid(148), stream=stream0)
        buf376 = empty_strided_cuda((4, 1, 74), (74, 74, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_253], Original ATen: [aten.reflection_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad1d_55.run(primals_23, buf376, 296, grid=grid(296), stream=stream0)
        # Topologically Sorted Source Nodes: [conv2d_90], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(reinterpret_tensor(buf376, (4, 1, 2, 37), (74, 0, 37, 1), 0), buf348, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf377, (4, 32, 1, 37), (1184, 37, 37, 1))
        buf378 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [conv2d_90, x_256], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_56.run(buf378, primals_152, 4736, grid=grid(4736), stream=stream0)
        del primals_152
        # Topologically Sorted Source Nodes: [conv2d_91], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, buf353, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 128, 1, 37), (4736, 37, 37, 1))
        buf380 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [conv2d_91, x_258], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_57.run(buf380, primals_155, 18944, grid=grid(18944), stream=stream0)
        del primals_155
        # Topologically Sorted Source Nodes: [conv2d_92], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, buf358, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 512, 1, 37), (18944, 37, 37, 1))
        buf382 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [conv2d_92, x_260], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_58.run(buf382, primals_158, 75776, grid=grid(75776), stream=stream0)
        del primals_158
        # Topologically Sorted Source Nodes: [conv2d_93], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, buf363, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 1024, 1, 37), (37888, 37, 37, 1))
        buf384 = buf383; del buf383  # reuse
        # Topologically Sorted Source Nodes: [conv2d_93, x_262], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf384, primals_161, 151552, grid=grid(151552), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [conv2d_94], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, buf368, stride=(3, 1), padding=(2, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 1024, 1, 37), (37888, 37, 37, 1))
        buf386 = buf385; del buf385  # reuse
        # Topologically Sorted Source Nodes: [conv2d_94, x_264], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_59.run(buf386, primals_164, 151552, grid=grid(151552), stream=stream0)
        del primals_164
        # Topologically Sorted Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf386, buf373, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (4, 1, 1, 37), (37, 37, 37, 1))
        buf388 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [x_266], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_60.run(buf388, primals_167, 148, grid=grid(148), stream=stream0)
        del primals_167
    return (reinterpret_tensor(buf34, (4, 1), (1, 1), 0), reinterpret_tensor(buf77, (4, 2), (2, 1), 0), reinterpret_tensor(buf119, (4, 3), (3, 1), 0), reinterpret_tensor(buf162, (4, 5), (5, 1), 0), reinterpret_tensor(buf205, (4, 7), (7, 1), 0), reinterpret_tensor(buf247, (4, 11), (11, 1), 0), reinterpret_tensor(buf289, (4, 17), (17, 1), 0), reinterpret_tensor(buf332, (4, 23), (23, 1), 0), reinterpret_tensor(buf375, (4, 37), (37, 1), 0), reinterpret_tensor(buf48, (4, 1), (1, 1), 0), reinterpret_tensor(buf89, (4, 2), (2, 1), 0), reinterpret_tensor(buf132, (4, 3), (3, 1), 0), reinterpret_tensor(buf175, (4, 5), (5, 1), 0), reinterpret_tensor(buf218, (4, 7), (7, 1), 0), reinterpret_tensor(buf259, (4, 11), (11, 1), 0), reinterpret_tensor(buf302, (4, 17), (17, 1), 0), reinterpret_tensor(buf345, (4, 23), (23, 1), 0), reinterpret_tensor(buf388, (4, 37), (37, 1), 0), buf4, buf9, buf14, buf19, buf24, buf29, buf34, buf52, buf57, buf62, buf67, buf72, buf77, buf94, buf99, buf104, buf109, buf114, buf119, buf137, buf142, buf147, buf152, buf157, buf162, buf180, buf185, buf190, buf195, buf200, buf205, buf222, buf227, buf232, buf237, buf242, buf247, buf264, buf269, buf274, buf279, buf284, buf289, buf307, buf312, buf317, buf322, buf327, buf332, buf350, buf355, buf360, buf365, buf370, buf375, buf36, buf38, buf40, buf42, buf44, buf46, buf48, buf79, buf81, buf83, buf85, buf87, buf89, buf122, buf124, buf126, buf128, buf130, buf132, buf165, buf167, buf169, buf171, buf173, buf175, buf208, buf210, buf212, buf214, buf216, buf218, buf249, buf251, buf253, buf255, buf257, buf259, buf292, buf294, buf296, buf298, buf300, buf302, buf335, buf337, buf339, buf341, buf343, buf345, buf378, buf380, buf382, buf384, buf386, buf388, primals_1, primals_2, primals_4, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, buf1, buf2, buf4, buf6, buf7, buf9, buf11, buf12, buf14, buf16, buf17, buf19, buf21, buf22, buf24, buf26, buf27, buf29, buf31, buf32, buf36, buf38, buf40, buf42, buf44, buf46, buf49, buf50, buf52, buf54, buf55, buf57, buf59, buf60, buf62, buf64, buf65, buf67, buf69, buf70, buf72, buf74, buf75, buf79, buf81, buf83, buf85, buf87, buf90, buf91, buf92, buf94, buf96, buf97, buf99, buf101, buf102, buf104, buf106, buf107, buf109, buf111, buf112, buf114, buf116, buf117, buf120, buf122, buf124, buf126, buf128, buf130, reinterpret_tensor(buf133, (4, 1, 13, 5), (65, 65, 5, 1), 0), buf134, buf135, buf137, buf139, buf140, buf142, buf144, buf145, buf147, buf149, buf150, buf152, buf154, buf155, buf157, buf159, buf160, reinterpret_tensor(buf163, (4, 1, 13, 5), (65, 65, 5, 1), 0), buf165, buf167, buf169, buf171, buf173, reinterpret_tensor(buf176, (4, 1, 10, 7), (70, 70, 7, 1), 0), buf177, buf178, buf180, buf182, buf183, buf185, buf187, buf188, buf190, buf192, buf193, buf195, buf197, buf198, buf200, buf202, buf203, reinterpret_tensor(buf206, (4, 1, 10, 7), (70, 70, 7, 1), 0), buf208, buf210, buf212, buf214, buf216, buf219, buf220, buf222, buf224, buf225, buf227, buf229, buf230, buf232, buf234, buf235, buf237, buf239, buf240, buf242, buf244, buf245, buf249, buf251, buf253, buf255, buf257, reinterpret_tensor(buf260, (4, 1, 4, 17), (68, 68, 17, 1), 0), buf261, buf262, buf264, buf266, buf267, buf269, buf271, buf272, buf274, buf276, buf277, buf279, buf281, buf282, buf284, buf286, buf287, reinterpret_tensor(buf290, (4, 1, 4, 17), (68, 68, 17, 1), 0), buf292, buf294, buf296, buf298, buf300, reinterpret_tensor(buf303, (4, 1, 3, 23), (69, 69, 23, 1), 0), buf304, buf305, buf307, buf309, buf310, buf312, buf314, buf315, buf317, buf319, buf320, buf322, buf324, buf325, buf327, buf329, buf330, reinterpret_tensor(buf333, (4, 1, 3, 23), (69, 69, 23, 1), 0), buf335, buf337, buf339, buf341, buf343, reinterpret_tensor(buf346, (4, 1, 2, 37), (74, 74, 37, 1), 0), buf347, buf348, buf350, buf352, buf353, buf355, buf357, buf358, buf360, buf362, buf363, buf365, buf367, buf368, buf370, buf372, buf373, reinterpret_tensor(buf376, (4, 1, 2, 37), (74, 74, 37, 1), 0), buf378, buf380, buf382, buf384, buf386, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 15), (15, 15, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1024, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1024, 4, 41), (164, 41, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1024, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1024, 1024, 5), (5120, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, 1024, 3), (3072, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, 1, 64), (64, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((32, 1, 5, 1), (5, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, 32, 5, 1), (160, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, 128, 5, 1), (640, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, 512, 5, 1), (2560, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, 1024, 5, 1), (5120, 5, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1, 1024, 3, 1), (3072, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
