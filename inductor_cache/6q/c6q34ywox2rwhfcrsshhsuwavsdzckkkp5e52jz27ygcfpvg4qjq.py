# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/ja/cjax6rbkththhotjxchjit6raesjusu64hbqlonngvbxghydtk3b.py
# Topologically Sorted Source Nodes: [hidden], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hidden => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_2, %primals_1], 1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (4*x1 + ((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ag/cagxyfw7xj7wjqlaclgnqrwerljwysa6g5dospjqgzx7dvjxhsn4.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_2 => relu
#   input_3 => add, add_1, mul, mul_1, mul_2, reciprocal, sqrt, sub
#   input_4 => relu_1
# Graph fragment:
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%addmm,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_6, 1e-05), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%sqrt,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%relu, %primals_5), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %mul), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_7), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_8), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = triton_helpers.maximum(tmp1, tmp17)
    tl.store(out_ptr0 + (x2), tmp18, None)
''', device_str='cuda')


# kernel path: inductor_cache/e6/ce6uy6qjjht6qgobkck5x2x3t3yyoocrp7kx3touhw5dmsm565hq.py
# Topologically Sorted Source Nodes: [input_37, input_38], Original ATen: [aten.addmm, aten.sigmoid]
# Source node to ATen node mapping:
#   input_37 => add_tensor
#   input_38 => sigmoid
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_58), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_tensor,), kwargs = {})
triton_poi_fused_addmm_sigmoid_2 = async_compile.triton('triton_poi_fused_addmm_sigmoid_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_sigmoid_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_sigmoid_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp4 = tl.sigmoid(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (2048, 8), (8, 1))
    assert_size_stride(primals_4, (2048, ), (1, ))
    assert_size_stride(primals_5, (2048, ), (1, ))
    assert_size_stride(primals_6, (2048, ), (1, ))
    assert_size_stride(primals_7, (2048, ), (1, ))
    assert_size_stride(primals_8, (2048, ), (1, ))
    assert_size_stride(primals_9, (2048, 2048), (2048, 1))
    assert_size_stride(primals_10, (2048, ), (1, ))
    assert_size_stride(primals_11, (2048, ), (1, ))
    assert_size_stride(primals_12, (2048, ), (1, ))
    assert_size_stride(primals_13, (2048, ), (1, ))
    assert_size_stride(primals_14, (2048, ), (1, ))
    assert_size_stride(primals_15, (2048, 2048), (2048, 1))
    assert_size_stride(primals_16, (2048, ), (1, ))
    assert_size_stride(primals_17, (2048, ), (1, ))
    assert_size_stride(primals_18, (2048, ), (1, ))
    assert_size_stride(primals_19, (2048, ), (1, ))
    assert_size_stride(primals_20, (2048, ), (1, ))
    assert_size_stride(primals_21, (2048, 2048), (2048, 1))
    assert_size_stride(primals_22, (2048, ), (1, ))
    assert_size_stride(primals_23, (2048, ), (1, ))
    assert_size_stride(primals_24, (2048, ), (1, ))
    assert_size_stride(primals_25, (2048, ), (1, ))
    assert_size_stride(primals_26, (2048, ), (1, ))
    assert_size_stride(primals_27, (2048, 2048), (2048, 1))
    assert_size_stride(primals_28, (2048, ), (1, ))
    assert_size_stride(primals_29, (2048, ), (1, ))
    assert_size_stride(primals_30, (2048, ), (1, ))
    assert_size_stride(primals_31, (2048, ), (1, ))
    assert_size_stride(primals_32, (2048, ), (1, ))
    assert_size_stride(primals_33, (2048, 2048), (2048, 1))
    assert_size_stride(primals_34, (2048, ), (1, ))
    assert_size_stride(primals_35, (2048, ), (1, ))
    assert_size_stride(primals_36, (2048, ), (1, ))
    assert_size_stride(primals_37, (2048, ), (1, ))
    assert_size_stride(primals_38, (2048, ), (1, ))
    assert_size_stride(primals_39, (2048, 2048), (2048, 1))
    assert_size_stride(primals_40, (2048, ), (1, ))
    assert_size_stride(primals_41, (2048, ), (1, ))
    assert_size_stride(primals_42, (2048, ), (1, ))
    assert_size_stride(primals_43, (2048, ), (1, ))
    assert_size_stride(primals_44, (2048, ), (1, ))
    assert_size_stride(primals_45, (2048, 2048), (2048, 1))
    assert_size_stride(primals_46, (2048, ), (1, ))
    assert_size_stride(primals_47, (2048, ), (1, ))
    assert_size_stride(primals_48, (2048, ), (1, ))
    assert_size_stride(primals_49, (2048, ), (1, ))
    assert_size_stride(primals_50, (2048, ), (1, ))
    assert_size_stride(primals_51, (2048, 2048), (2048, 1))
    assert_size_stride(primals_52, (2048, ), (1, ))
    assert_size_stride(primals_53, (2048, ), (1, ))
    assert_size_stride(primals_54, (2048, ), (1, ))
    assert_size_stride(primals_55, (2048, ), (1, ))
    assert_size_stride(primals_56, (2048, ), (1, ))
    assert_size_stride(primals_57, (1, 2048), (2048, 1))
    assert_size_stride(primals_58, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_2, primals_1, buf0, 32, grid=grid(32), stream=stream0)
        del primals_1
        del primals_2
        buf1 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_4, buf0, reinterpret_tensor(primals_3, (8, 2048), (1, 8), 0), alpha=1, beta=1, out=buf1)
        del primals_3
        del primals_4
        buf2 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf1, primals_5, primals_6, primals_7, primals_8, buf2, 8192, grid=grid(8192), stream=stream0)
        del primals_8
        buf3 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, buf2, reinterpret_tensor(primals_9, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf3)
        del primals_10
        buf4 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7, input_8], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf3, primals_11, primals_12, primals_13, primals_14, buf4, 8192, grid=grid(8192), stream=stream0)
        del primals_14
        buf5 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_16, buf4, reinterpret_tensor(primals_15, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf5)
        del primals_16
        buf6 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf5, primals_17, primals_18, primals_19, primals_20, buf6, 8192, grid=grid(8192), stream=stream0)
        del primals_20
        buf7 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_22, buf6, reinterpret_tensor(primals_21, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf7)
        del primals_22
        buf8 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15, input_16], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf7, primals_23, primals_24, primals_25, primals_26, buf8, 8192, grid=grid(8192), stream=stream0)
        del primals_26
        buf9 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_28, buf8, reinterpret_tensor(primals_27, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf9)
        del primals_28
        buf10 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19, input_20], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf9, primals_29, primals_30, primals_31, primals_32, buf10, 8192, grid=grid(8192), stream=stream0)
        del primals_32
        buf11 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_34, buf10, reinterpret_tensor(primals_33, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf11)
        del primals_34
        buf12 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23, input_24], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf11, primals_35, primals_36, primals_37, primals_38, buf12, 8192, grid=grid(8192), stream=stream0)
        del primals_38
        buf13 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_40, buf12, reinterpret_tensor(primals_39, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf13)
        del primals_40
        buf14 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27, input_28], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf13, primals_41, primals_42, primals_43, primals_44, buf14, 8192, grid=grid(8192), stream=stream0)
        del primals_44
        buf15 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_46, buf14, reinterpret_tensor(primals_45, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf15)
        del primals_46
        buf16 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_30, input_31, input_32], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf15, primals_47, primals_48, primals_49, primals_50, buf16, 8192, grid=grid(8192), stream=stream0)
        del primals_50
        buf17 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_52, buf16, reinterpret_tensor(primals_51, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf17)
        del primals_52
        buf18 = empty_strided_cuda((4, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.relu, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf17, primals_53, primals_54, primals_55, primals_56, buf18, 8192, grid=grid(8192), stream=stream0)
        del primals_56
        buf19 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.addmm]
        extern_kernels.mm(buf18, reinterpret_tensor(primals_57, (2048, 1), (1, 2048), 0), out=buf19)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_37, input_38], Original ATen: [aten.addmm, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_sigmoid_2.run(buf20, primals_58, 4, grid=grid(4), stream=stream0)
        del primals_58
    return (buf20, primals_5, primals_6, primals_7, primals_11, primals_12, primals_13, primals_17, primals_18, primals_19, primals_23, primals_24, primals_25, primals_29, primals_30, primals_31, primals_35, primals_36, primals_37, primals_41, primals_42, primals_43, primals_47, primals_48, primals_49, primals_53, primals_54, primals_55, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf20, primals_57, primals_51, primals_45, primals_39, primals_33, primals_27, primals_21, primals_15, primals_9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2048, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
