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


# kernel path: inductor_cache/nl/cnly6asv5srtktfefsr6f3dxpeky5vwzyjg65inq2gb624ird425.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_0 = async_compile.triton('triton_poi_fused_convolution_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/3t/c3t7hqey5kjqpvj6kynz4bmaefpv3tlw6i65qxswbjaji3e7unjf.py
# Topologically Sorted Source Nodes: [input_3, y], Original ATen: [aten.convolution, aten.mean]
# Source node to ATen node mapping:
#   input_3 => convolution_1
#   y => mean
# Graph fragment:
#   %convolution_1 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_4, %primals_5, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_1, [-1, -2], True), kwargs = {})
triton_red_fused_convolution_mean_1 = async_compile.triton('triton_red_fused_convolution_mean_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_1', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_1(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = (xindex % 64)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + 4096*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
        tl.store(in_out_ptr0 + (r2 + 4096*x3), tmp2, rmask & xmask)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2y/c2ysrdvmby55mp53fcbhlg42w2f6rhdhktce6xnzrkxi2eblw6zy.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_4 => convolution_2
#   input_5 => relu_1
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_6, %primals_7, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_poi_fused_convolution_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oo/cooeljrzyt6cszyduj5h2s2rbrghfb7efrcgiwbqvwqbxsoopw6y.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_6 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kd/ckdmpowvchddfixh3yuzk7gjskzanj6ahtn7grvaspv5xhi5atuc.py
# Topologically Sorted Source Nodes: [input_7, res, input_8], Original ATen: [aten.sigmoid, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_7 => sigmoid
#   input_8 => add
#   res => mul
# Graph fragment:
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_3,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_1, %sigmoid), kwargs = {})
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %primals_3), kwargs = {})
triton_poi_fused_add_mul_sigmoid_4 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 4096
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp5 = tmp3 + tmp4
    tl.store(out_ptr0 + (x2), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/qq/cqqasysjbjbdm4qvc4mzc75kszpluaovosgzafp3zvpikknpsgfk.py
# Topologically Sorted Source Nodes: [input_97, add_12], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   add_12 => add_12
#   input_97 => convolution_48
# Graph fragment:
#   %convolution_48 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_11, %primals_98, %primals_99, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_48, %primals_3), kwargs = {})
triton_poi_fused_add_convolution_5 = async_compile.triton('triton_poi_fused_add_convolution_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99 = args
    args.clear()
    assert_size_stride(primals_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 64, 64, 64), (262144, 4096, 64, 1))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_31, (4, ), (1, ))
    assert_size_stride(primals_32, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_33, (64, ), (1, ))
    assert_size_stride(primals_34, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_39, (4, ), (1, ))
    assert_size_stride(primals_40, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_41, (64, ), (1, ))
    assert_size_stride(primals_42, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_43, (64, ), (1, ))
    assert_size_stride(primals_44, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_47, (4, ), (1, ))
    assert_size_stride(primals_48, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_49, (64, ), (1, ))
    assert_size_stride(primals_50, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_55, (4, ), (1, ))
    assert_size_stride(primals_56, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_61, (64, ), (1, ))
    assert_size_stride(primals_62, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_63, (4, ), (1, ))
    assert_size_stride(primals_64, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_71, (4, ), (1, ))
    assert_size_stride(primals_72, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (4, ), (1, ))
    assert_size_stride(primals_80, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_87, (4, ), (1, ))
    assert_size_stride(primals_88, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (4, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_95, (4, ), (1, ))
    assert_size_stride(primals_96, (64, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_97, (64, ), (1, ))
    assert_size_stride(primals_98, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_99, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf1, primals_2, 1048576, grid=grid(1048576), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf5 = reinterpret_tensor(buf4, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_3, y], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf3, buf5, primals_5, 256, 4096, grid=grid(256), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 1, 1), (4, 1, 1, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf7, primals_7, 16, grid=grid(16), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 64, 1, 1), (64, 1, 1, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf9, primals_9, 256, grid=grid(256), stream=stream0)
        del primals_9
        buf10 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, res, input_8], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf3, buf9, primals_3, buf10, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf12 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf12, primals_11, 1048576, grid=grid(1048576), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf16 = reinterpret_tensor(buf15, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [input_11, y_1], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf14, buf16, primals_13, 256, 4096, grid=grid(256), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 4, 1, 1), (4, 1, 1, 1))
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf18, primals_15, 16, grid=grid(16), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 64, 1, 1), (64, 1, 1, 1))
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf20, primals_17, 256, grid=grid(256), stream=stream0)
        del primals_17
        buf21 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15, res_1, input_16], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf14, buf20, buf10, buf21, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf23 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_17, input_18], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf23, primals_19, 1048576, grid=grid(1048576), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf25 = buf24; del buf24  # reuse
        buf26 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf27 = reinterpret_tensor(buf26, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [input_19, y_2], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf25, buf27, primals_21, 256, 4096, grid=grid(256), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 4, 1, 1), (4, 1, 1, 1))
        buf29 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf29, primals_23, 16, grid=grid(16), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 1, 1), (64, 1, 1, 1))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf31, primals_25, 256, grid=grid(256), stream=stream0)
        del primals_25
        buf32 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, res_2, input_24], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf25, buf31, buf21, buf32, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_25, input_26], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf34, primals_27, 1048576, grid=grid(1048576), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf38 = reinterpret_tensor(buf37, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [input_27, y_3], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf36, buf38, primals_29, 256, 4096, grid=grid(256), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 4, 1, 1), (4, 1, 1, 1))
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf40, primals_31, 16, grid=grid(16), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 64, 1, 1), (64, 1, 1, 1))
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf42, primals_33, 256, grid=grid(256), stream=stream0)
        del primals_33
        buf43 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31, res_3, input_32], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf36, buf42, buf32, buf43, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf45, primals_35, 1048576, grid=grid(1048576), stream=stream0)
        del primals_35
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_36, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf47 = buf46; del buf46  # reuse
        buf48 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf49 = reinterpret_tensor(buf48, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [input_35, y_4], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf47, buf49, primals_37, 256, 4096, grid=grid(256), stream=stream0)
        del primals_37
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 4, 1, 1), (4, 1, 1, 1))
        buf51 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf51, primals_39, 16, grid=grid(16), stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 1, 1), (64, 1, 1, 1))
        buf53 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf53, primals_41, 256, grid=grid(256), stream=stream0)
        del primals_41
        buf54 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, res_4, input_40], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf47, buf53, buf43, buf54, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf56 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf56, primals_43, 1048576, grid=grid(1048576), stream=stream0)
        del primals_43
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf58 = buf57; del buf57  # reuse
        buf59 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf60 = reinterpret_tensor(buf59, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_43, y_5], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf58, buf60, primals_45, 256, 4096, grid=grid(256), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 4, 1, 1), (4, 1, 1, 1))
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf62, primals_47, 16, grid=grid(16), stream=stream0)
        del primals_47
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_48, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 64, 1, 1), (64, 1, 1, 1))
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf64, primals_49, 256, grid=grid(256), stream=stream0)
        del primals_49
        buf65 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_47, res_5, input_48], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf58, buf64, buf54, buf65, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf67, primals_51, 1048576, grid=grid(1048576), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf69 = buf68; del buf68  # reuse
        buf70 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf71 = reinterpret_tensor(buf70, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_51, y_6], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf69, buf71, primals_53, 256, 4096, grid=grid(256), stream=stream0)
        del primals_53
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_54, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 4, 1, 1), (4, 1, 1, 1))
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf73, primals_55, 16, grid=grid(16), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 64, 1, 1), (64, 1, 1, 1))
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf75, primals_57, 256, grid=grid(256), stream=stream0)
        del primals_57
        buf76 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_55, res_6, input_56], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf69, buf75, buf65, buf76, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf78, primals_59, 1048576, grid=grid(1048576), stream=stream0)
        del primals_59
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf80 = buf79; del buf79  # reuse
        buf81 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf82 = reinterpret_tensor(buf81, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [input_59, y_7], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf80, buf82, primals_61, 256, 4096, grid=grid(256), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 4, 1, 1), (4, 1, 1, 1))
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf84, primals_63, 16, grid=grid(16), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 64, 1, 1), (64, 1, 1, 1))
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf86, primals_65, 256, grid=grid(256), stream=stream0)
        del primals_65
        buf87 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_63, res_7, input_64], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf80, buf86, buf76, buf87, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [input_65, input_66], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf89, primals_67, 1048576, grid=grid(1048576), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf91 = buf90; del buf90  # reuse
        buf92 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf93 = reinterpret_tensor(buf92, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [input_67, y_8], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf91, buf93, primals_69, 256, 4096, grid=grid(256), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 4, 1, 1), (4, 1, 1, 1))
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf95, primals_71, 16, grid=grid(16), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 64, 1, 1), (64, 1, 1, 1))
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf97, primals_73, 256, grid=grid(256), stream=stream0)
        del primals_73
        buf98 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, res_8, input_72], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf91, buf97, buf87, buf98, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf100, primals_75, 1048576, grid=grid(1048576), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf102 = buf101; del buf101  # reuse
        buf103 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf104 = reinterpret_tensor(buf103, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [input_75, y_9], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf102, buf104, primals_77, 256, 4096, grid=grid(256), stream=stream0)
        del primals_77
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 4, 1, 1), (4, 1, 1, 1))
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf106, primals_79, 16, grid=grid(16), stream=stream0)
        del primals_79
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 64, 1, 1), (64, 1, 1, 1))
        buf108 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf108, primals_81, 256, grid=grid(256), stream=stream0)
        del primals_81
        buf109 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_79, res_9, input_80], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf102, buf108, buf98, buf109, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_81], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_81, input_82], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf111, primals_83, 1048576, grid=grid(1048576), stream=stream0)
        del primals_83
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf115 = reinterpret_tensor(buf114, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf114  # reuse
        # Topologically Sorted Source Nodes: [input_83, y_10], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf113, buf115, primals_85, 256, 4096, grid=grid(256), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 4, 1, 1), (4, 1, 1, 1))
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf117, primals_87, 16, grid=grid(16), stream=stream0)
        del primals_87
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 64, 1, 1), (64, 1, 1, 1))
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf119, primals_89, 256, grid=grid(256), stream=stream0)
        del primals_89
        buf120 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_87, res_10, input_88], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf113, buf119, buf109, buf120, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_0.run(buf122, primals_91, 1048576, grid=grid(1048576), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf124 = buf123; del buf123  # reuse
        buf125 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf126 = reinterpret_tensor(buf125, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf125  # reuse
        # Topologically Sorted Source Nodes: [input_91, y_11], Original ATen: [aten.convolution, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_1.run(buf124, buf126, primals_93, 256, 4096, grid=grid(256), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [input_92], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 4, 1, 1), (4, 1, 1, 1))
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_2.run(buf128, primals_95, 16, grid=grid(16), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_96, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 64, 1, 1), (64, 1, 1, 1))
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf130, primals_97, 256, grid=grid(256), stream=stream0)
        del primals_97
        buf131 = empty_strided_cuda((4, 64, 64, 64), (262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_95, res_11, input_96], Original ATen: [aten.sigmoid, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_4.run(buf124, buf130, buf120, buf131, 1048576, grid=grid(1048576), stream=stream0)
        # Topologically Sorted Source Nodes: [input_97], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 64, 64, 64), (262144, 4096, 64, 1))
        buf133 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [input_97, add_12], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_5.run(buf133, primals_99, primals_3, 1048576, grid=grid(1048576), stream=stream0)
        del primals_99
    return (buf133, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, buf1, buf3, buf5, buf7, buf9, buf10, buf12, buf14, buf16, buf18, buf20, buf21, buf23, buf25, buf27, buf29, buf31, buf32, buf34, buf36, buf38, buf40, buf42, buf43, buf45, buf47, buf49, buf51, buf53, buf54, buf56, buf58, buf60, buf62, buf64, buf65, buf67, buf69, buf71, buf73, buf75, buf76, buf78, buf80, buf82, buf84, buf86, buf87, buf89, buf91, buf93, buf95, buf97, buf98, buf100, buf102, buf104, buf106, buf108, buf109, buf111, buf113, buf115, buf117, buf119, buf120, buf122, buf124, buf126, buf128, buf130, buf131, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 64, 64, 64), (262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((4, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
