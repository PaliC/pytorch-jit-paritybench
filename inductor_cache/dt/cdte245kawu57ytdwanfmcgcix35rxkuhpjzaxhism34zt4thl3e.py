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


# kernel path: inductor_cache/qg/cqgureo7lmnwshqfkuf2dy74kgptoryxsbpxbk3hc3qugdynkqjk.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => _unsafe_index, _unsafe_index_1
#   input_3 => convolution_1
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_reflection_pad2d_0 = async_compile.triton('triton_poi_fused_convolution_reflection_pad2d_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_reflection_pad2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_reflection_pad2d_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x4 = xindex // 4356
    x2 = ((xindex // 4356) % 3)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x5), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4n/c4nsmbgu2g6q5elcrt3s7couvmcf3wumlow7ms4tsyldmql6eiho.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.reflection_pad2d, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => _unsafe_index, _unsafe_index_1
#   input_3 => convolution_1
#   input_4 => relu
#   input_5 => _unsafe_index_2, _unsafe_index_3
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%relu, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_3 : [num_users=3] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_2, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_convolution_reflection_pad2d_relu_1 = async_compile.triton('triton_poi_fused_convolution_reflection_pad2d_relu_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_reflection_pad2d_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_reflection_pad2d_relu_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1115136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x4 = xindex // 4356
    x2 = ((xindex // 4356) % 64)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(out_ptr0 + (x5), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/45/c45fqgy5jgdxf7cxrpgo2bbqbecioaqjkmyabwnqcq4eo3kr3h3x.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   input_7 => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_red_fused__native_batch_norm_legit_2 = async_compile.triton('triton_red_fused__native_batch_norm_legit_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_2(in_out_ptr0, in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: inductor_cache/qp/cqptd4k5hoh6gv5apkidqlzaxd6svnty5yb7swh4cn456ygkayfp.py
# Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_9 => _unsafe_index_4, _unsafe_index_5
# Graph fragment:
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%view_3, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_4, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_3 = async_compile.triton('triton_poi_fused_reflection_pad2d_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1115136
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
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ti/ctiry7hk2om52hrmo4ber4abcc7blkwfxjqs2fuzbjjzxj6vyfam.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_11, out, input_33, out_3], Original ATen: [aten.convolution, aten.reflection_pad2d, aten.relu, aten._native_batch_norm_legit, aten.add]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_11 => add_1, rsqrt_1, var_mean_1
#   input_2 => _unsafe_index, _unsafe_index_1
#   input_3 => convolution_1
#   input_33 => add_11, rsqrt_8, var_mean_8
#   input_4 => relu
#   out => add_2
#   out_3 => add_12
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index, [None, None, None, %sub_1]), kwargs = {})
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_1, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_1,), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %view_6), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_28, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_16, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_11,), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, %view_29), kwargs = {})
triton_red_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_4 = async_compile.triton('triton_red_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr2, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp7_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr1 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp7_mean_next, tmp7_m2_next, tmp7_weight_next = triton_helpers.welford_reduce(
            tmp6, tmp7_mean, tmp7_m2, tmp7_weight, roffset == 0
        )
        tmp7_mean = tl.where(rmask & xmask, tmp7_mean_next, tmp7_mean)
        tmp7_m2 = tl.where(rmask & xmask, tmp7_m2_next, tmp7_m2)
        tmp7_weight = tl.where(rmask & xmask, tmp7_weight_next, tmp7_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp7_tmp, tmp8_tmp, tmp9_tmp = triton_helpers.welford(
        tmp7_mean, tmp7_m2, tmp7_weight, 1
    )
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tl.store(out_ptr2 + (x0), tmp7, xmask)
    x2 = (xindex % 64)
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_ptr2 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tl.full([1, 1], 0, tl.int32)
        tmp14 = triton_helpers.maximum(tmp13, tmp12)
        tmp16 = tmp15 - tmp7
        tmp17 = 4096.0
        tmp18 = tmp8 / tmp17
        tmp19 = 1e-05
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp14 + tmp22
        tmp25 = tmp24 - tmp2
        tmp26 = tmp3 / tmp17
        tmp27 = tmp26 + tmp19
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp25 * tmp28
        tmp30 = tmp14 + tmp29
        tl.store(out_ptr4 + (r1 + 4096*x0), tmp23, rmask & xmask)
        tl.store(out_ptr5 + (r1 + 4096*x0), tmp30, rmask & xmask)
    tmp31 = 4096.0
    tmp32 = tmp8 / tmp31
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp3 / tmp31
    tmp37 = tmp36 + tmp33
    tmp38 = libdevice.rsqrt(tmp37)
    tl.store(out_ptr6 + (x0), tmp35, xmask)
    tl.store(out_ptr7 + (x0), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ma/cmab6q66qu72s4atpcqclvrfpfy4ghp2bgteq5euzvu5ntreso6g.py
# Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_12 => _unsafe_index_6, _unsafe_index_7
# Graph fragment:
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_2, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_7 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_6, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_reflection_pad2d_5 = async_compile.triton('triton_poi_fused_reflection_pad2d_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_reflection_pad2d_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_reflection_pad2d_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1115136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x2 = xindex // 4356
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i3/ci3jguc524n2tdhekoejkjbpprqvqtiudyvszwps4eqm44e7nexu.py
# Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   input_18 => add_4, rsqrt_3, var_mean_3
# Graph fragment:
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
triton_red_fused__native_batch_norm_legit_6 = async_compile.triton('triton_red_fused__native_batch_norm_legit_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_6(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: inductor_cache/sx/csxzielijpamyijeorbwsxuy6wn6elavaiusnkibk6hmnojfvfm6.py
# Topologically Sorted Source Nodes: [out_1, input_19], Original ATen: [aten.add, aten.reflection_pad2d]
# Source node to ATen node mapping:
#   input_19 => _unsafe_index_10, _unsafe_index_11
#   out_1 => add_5
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_13), kwargs = {})
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%add_5, [None, None, %sub_1, None]), kwargs = {})
#   %_unsafe_index_11 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_unsafe_index_10, [None, None, None, %sub_1]), kwargs = {})
triton_poi_fused_add_reflection_pad2d_7 = async_compile.triton('triton_poi_fused_add_reflection_pad2d_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_reflection_pad2d_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_reflection_pad2d_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1115136
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 66)
    x1 = ((xindex // 66) % 66)
    x2 = xindex // 4356
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4095 + ((-1)*tl_math.abs((-63) + tl_math.abs((-1) + x0))) + ((-64)*tl_math.abs((-63) + tl_math.abs((-1) + x1))) + 4096*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 4096.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp11 = tmp0 + tmp10
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g3/cg3qhf5b2ucqdz3uakwh5fawqva3lss53vxcfhmogmbse27dc3rr.py
# Topologically Sorted Source Nodes: [out_1, input_25, out_2, input_26], Original ATen: [aten.add, aten._native_batch_norm_legit, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_25 => add_7, rsqrt_5, var_mean_5
#   input_26 => add_9, mul_6, rsqrt_6, sub_34, var_mean_6
#   out_1 => add_5
#   out_2 => add_8
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_13), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_19, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %view_20), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_21, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_21, %getitem_13), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %rsqrt_6), kwargs = {})
#   %sub_114 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_21, %unsqueeze_86), kwargs = {})
triton_red_fused__native_batch_norm_legit_add_native_batch_norm_backward_8 = async_compile.triton('triton_red_fused__native_batch_norm_legit_add_native_batch_norm_backward_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 4096},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_add_native_batch_norm_backward_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_add_native_batch_norm_backward_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tmp7 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp25_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_out_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr1 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tl.load(in_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tmp6 - tmp7
        tmp10 = 4096.0
        tmp11 = tmp9 / tmp10
        tmp12 = 1e-05
        tmp13 = tmp11 + tmp12
        tmp14 = libdevice.rsqrt(tmp13)
        tmp15 = tmp8 * tmp14
        tmp16 = tmp5 + tmp15
        tmp18 = tmp17 - tmp2
        tmp19 = tmp3 / tmp10
        tmp20 = tmp19 + tmp12
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp18 * tmp21
        tmp23 = tmp16 + tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp25_mean_next, tmp25_m2_next, tmp25_weight_next = triton_helpers.welford_reduce(
            tmp24, tmp25_mean, tmp25_m2, tmp25_weight, roffset == 0
        )
        tmp25_mean = tl.where(rmask & xmask, tmp25_mean_next, tmp25_mean)
        tmp25_m2 = tl.where(rmask & xmask, tmp25_m2_next, tmp25_m2)
        tmp25_weight = tl.where(rmask & xmask, tmp25_weight_next, tmp25_weight)
        tl.store(in_out_ptr0 + (r1 + 4096*x0), tmp23, rmask & xmask)
    tmp25_tmp, tmp26_tmp, tmp27_tmp = triton_helpers.welford(
        tmp25_mean, tmp25_m2, tmp25_weight, 1
    )
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp26_tmp[:, None]
    tmp27 = tmp27_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp28 = tl.load(in_out_ptr0 + (r1 + 4096*x0), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tmp28 - tmp25
        tmp30 = 4096.0
        tmp31 = tmp26 / tmp30
        tmp32 = 1e-05
        tmp33 = tmp31 + tmp32
        tmp34 = libdevice.rsqrt(tmp33)
        tmp35 = tmp29 * tmp34
        tl.store(out_ptr4 + (r1 + 4096*x0), tmp35, rmask & xmask)
        tl.store(out_ptr5 + (r1 + 4096*x0), tmp29, rmask & xmask)
    tmp36 = 4096.0
    tmp37 = tmp3 / tmp36
    tmp38 = 1e-05
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp41 = tmp26 / tmp36
    tmp42 = tmp41 + tmp38
    tmp43 = libdevice.rsqrt(tmp42)
    tl.store(out_ptr6 + (x0), tmp40, xmask)
    tl.store(out_ptr7 + (x0), tmp43, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (3, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (3, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_9, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_13, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_14, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_15, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_16, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_17, (64, 64, 3, 3), (576, 9, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 3, 64, 64), (12288, 4096, 64, 1))
        del primals_1
        del primals_3
        buf1 = empty_strided_cuda((4, 3, 66, 66), (13068, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_reflection_pad2d_0.run(buf0, primals_2, buf1, 52272, grid=grid(52272), stream=stream0)
        del buf0
        del primals_2
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten.reflection_pad2d]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 64, 64), (262144, 4096, 64, 1))
        del buf1
        del primals_4
        buf3 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_5], Original ATen: [aten.convolution, aten.reflection_pad2d, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_reflection_pad2d_relu_1.run(buf2, primals_5, buf3, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf5 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        buf6 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf8 = reinterpret_tensor(buf6, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_2.run(buf8, buf4, buf5, 256, 4096, grid=grid(256), stream=stream0)
        buf9 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_3.run(buf4, buf5, buf8, buf9, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 64, 64, 64), (262144, 4096, 64, 1))
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf3, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf47 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        buf48 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf50 = reinterpret_tensor(buf48, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_2.run(buf50, buf46, buf47, 256, 4096, grid=grid(256), stream=stream0)
        buf51 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_3.run(buf46, buf47, buf50, buf51, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf53 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf11 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf15 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        buf57 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        buf14 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf56 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, input_11, out, input_33, out_3], Original ATen: [aten.convolution, aten.reflection_pad2d, aten.relu, aten._native_batch_norm_legit, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_add_convolution_reflection_pad2d_relu_4.run(buf52, buf10, buf2, primals_5, buf53, buf11, buf15, buf57, buf14, buf56, 256, 4096, grid=grid(256), stream=stream0)
        del primals_5
        buf16 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf15, buf16, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf18 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        buf19 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf21 = reinterpret_tensor(buf19, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_2.run(buf21, buf17, buf18, 256, 4096, grid=grid(256), stream=stream0)
        buf22 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_3.run(buf17, buf18, buf21, buf22, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf24 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf25 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf27 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_6.run(buf23, buf24, buf25, buf27, 256, 4096, grid=grid(256), stream=stream0)
        buf28 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, input_19], Original ATen: [aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_reflection_pad2d_7.run(buf15, buf23, buf24, buf25, buf28, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf30 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        buf31 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf33 = reinterpret_tensor(buf31, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_2.run(buf33, buf29, buf30, 256, 4096, grid=grid(256), stream=stream0)
        buf34 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_3.run(buf29, buf30, buf33, buf34, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf36 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf40 = buf15; del buf15  # reuse
        buf45 = reinterpret_tensor(buf2, (1, 256, 64, 64), (1048576, 4096, 64, 1), 0); del buf2  # reuse
        buf89 = empty_strided_cuda((1, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32)
        buf39 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf44 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_1, input_25, out_2, input_26], Original ATen: [aten.add, aten._native_batch_norm_legit, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_add_native_batch_norm_backward_8.run(buf40, buf35, buf23, buf24, buf25, buf36, buf45, buf89, buf39, buf44, 256, 4096, grid=grid(256), stream=stream0)
        buf58 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_5.run(buf57, buf58, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf60 = reinterpret_tensor(buf25, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf25  # reuse
        buf61 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf63 = reinterpret_tensor(buf61, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_2.run(buf63, buf59, buf60, 256, 4096, grid=grid(256), stream=stream0)
        buf64 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_3.run(buf59, buf60, buf63, buf64, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf66 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf67 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf69 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_6.run(buf65, buf66, buf67, buf69, 256, 4096, grid=grid(256), stream=stream0)
        buf70 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, input_41], Original ATen: [aten.add, aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_reflection_pad2d_7.run(buf57, buf65, buf66, buf67, buf70, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf72 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        buf73 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf75 = reinterpret_tensor(buf73, (1, 256, 1, 1), (256, 1, 1, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_2.run(buf75, buf71, buf72, 256, 4096, grid=grid(256), stream=stream0)
        buf76 = empty_strided_cuda((4, 64, 66, 66), (278784, 4356, 66, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.reflection_pad2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_reflection_pad2d_3.run(buf71, buf72, buf75, buf76, 1115136, grid=grid(1115136), stream=stream0)
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf78 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf82 = buf57; del buf57  # reuse
        buf87 = reinterpret_tensor(buf40, (1, 256, 64, 64), (1048576, 4096, 64, 1), 0); del buf40  # reuse
        buf88 = empty_strided_cuda((1, 256, 64, 64), (1048576, 4096, 64, 1), torch.float32)
        buf81 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        buf86 = empty_strided_cuda((1, 256, 1, 1), (256, 1, 256, 256), torch.float32)
        # Topologically Sorted Source Nodes: [out_4, input_47, out_5, input_48], Original ATen: [aten.add, aten._native_batch_norm_legit, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__native_batch_norm_legit_add_native_batch_norm_backward_8.run(buf82, buf77, buf65, buf66, buf67, buf78, buf87, buf88, buf81, buf86, 256, 4096, grid=grid(256), stream=stream0)
        del buf67
        del buf82
    return (reinterpret_tensor(buf45, (4, 64, 64, 64), (262144, 4096, 64, 1), 0), reinterpret_tensor(buf87, (4, 64, 64, 64), (262144, 4096, 64, 1), 0), primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, buf3, buf4, buf5, buf8, buf9, buf10, reinterpret_tensor(buf14, (256, ), (1, ), 0), buf16, buf17, buf18, buf21, buf22, buf23, reinterpret_tensor(buf27, (256, ), (1, ), 0), buf28, buf29, buf30, buf33, buf34, buf35, reinterpret_tensor(buf39, (256, ), (1, ), 0), reinterpret_tensor(buf44, (256, ), (1, ), 0), buf46, buf47, buf50, buf51, buf52, reinterpret_tensor(buf56, (256, ), (1, ), 0), buf58, buf59, buf60, buf63, buf64, buf65, reinterpret_tensor(buf69, (256, ), (1, ), 0), buf70, buf71, buf72, buf75, buf76, buf77, reinterpret_tensor(buf81, (256, ), (1, ), 0), reinterpret_tensor(buf86, (256, ), (1, ), 0), buf88, reinterpret_tensor(buf78, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf66, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf53, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf89, reinterpret_tensor(buf36, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf11, (1, 256, 1, 1), (256, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
