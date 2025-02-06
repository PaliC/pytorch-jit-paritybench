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


# kernel path: inductor_cache/l2/cl2p652t4z33ayhwtx4ukqcdgnv46zzqfwms6mau6t67vcuic2qk.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_2 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze,), kwargs = {})
#   %le_7 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_relu_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr1 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ux/cuxhpizjrzxf2cln53rztmzacc3rcqxrsvgscfaaqoh4n66v6os7.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_4 => relu_1
# Graph fragment:
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_1,), kwargs = {})
#   %le_6 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_1, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_1 = async_compile.triton('triton_poi_fused_relu_threshold_backward_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_1(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vd/cvddsvgyyahutb2fp72aqbor3rzr4vfyec2ke62hfunzdbeo62gt.py
# Topologically Sorted Source Nodes: [mul, input_6], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_6 => add
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, 1.0), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %mul), kwargs = {})
triton_poi_fused_add_mul_2 = async_compile.triton('triton_poi_fused_add_mul_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uw/cuwonrfklvc6scs7e5kjr7j57rpwh7mom3ueemilfrv6gdli2nws.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_8 => relu_2
# Graph fragment:
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_3,), kwargs = {})
#   %le_5 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_3 = async_compile.triton('triton_poi_fused_relu_threshold_backward_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr1 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ck/cckxdeezv34eobm6puhowttz2tnk2bhg3uk3msqmk6qguu4npnnw.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_10 => relu_3
# Graph fragment:
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_4,), kwargs = {})
#   %le_4 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_3, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_4 = async_compile.triton('triton_poi_fused_relu_threshold_backward_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_4(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 80
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cd/ccd22qabiozrnkiimucofzrvhhwt6qehirdv4nzibjvwtetkasz7.py
# Topologically Sorted Source Nodes: [mul_1, input_12], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_12 => add_1
#   mul_1 => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, 1.0), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_3, %mul_1), kwargs = {})
triton_poi_fused_add_mul_5 = async_compile.triton('triton_poi_fused_add_mul_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hr/chram5fuaihe6aqmoaoijj5vjcmbwaoskvtptnhicpqbyh4d3ekr.py
# Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_14 => relu_4
# Graph fragment:
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_6,), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_4, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_6 = async_compile.triton('triton_poi_fused_relu_threshold_backward_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 6
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr1 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wk/cwk4sh4b5ahbdnj4b3lgsp5ekgs46446sqruibrwwy7l36beqfkc.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_16 => relu_5
# Graph fragment:
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_7,), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_5, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_7 = async_compile.triton('triton_poi_fused_relu_threshold_backward_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_7(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qa/cqakg7ye5622dhik3cfcoaxgcgbsakldbtyltbpr63xtlciplakt.py
# Topologically Sorted Source Nodes: [mul_2, input_18], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_18 => add_2
#   mul_2 => mul_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_8, 1.0), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_6, %mul_2), kwargs = {})
triton_poi_fused_add_mul_8 = async_compile.triton('triton_poi_fused_add_mul_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 6
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gr/cgrkexn3o452vgys4qh4rbyz6shh3jhmg6gi7jheqdf2aivk4ah5.py
# Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_20 => relu_6
# Graph fragment:
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_9,), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_6, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_9 = async_compile.triton('triton_poi_fused_relu_threshold_backward_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 7
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr1 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aw/cawuipzdvaop4o4wrkcb5pfpts7tvwlvyof4svfsrhgia32annl3.py
# Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_22 => relu_7
# Graph fragment:
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%squeeze_10,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_7, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_10 = async_compile.triton('triton_poi_fused_relu_threshold_backward_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_10(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 7
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6z/c6zwpgpg7r7bpdxw55glnktnkpkczrhiu27kie555mlqc4pgdblw.py
# Topologically Sorted Source Nodes: [mul_3, input_24], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_24 => add_3
#   mul_3 => mul_3
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_11, 1.0), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_9, %mul_3), kwargs = {})
triton_poi_fused_add_mul_11 = async_compile.triton('triton_poi_fused_add_mul_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 7
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp2 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lo/clornuunwnr45jd62rfdspmdsizquysf5mcx3tz6dhokazcxwam2.py
# Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_25 => convolution_12
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%unsqueeze_12, %primals_26, %primals_27, [1], [0], [1], True, [0], 1), kwargs = {})
triton_poi_fused_convolution_12 = async_compile.triton('triton_poi_fused_convolution_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 8
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3), (12, 3, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (16, 4, 3), (12, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (4, 16, 1), (16, 1, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 2), (8, 2, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (16, 4, 3), (12, 3, 1))
    assert_size_stride(primals_11, (16, ), (1, ))
    assert_size_stride(primals_12, (4, 16, 1), (16, 1, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 2), (8, 2, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (16, 4, 3), (12, 3, 1))
    assert_size_stride(primals_17, (16, ), (1, ))
    assert_size_stride(primals_18, (4, 16, 1), (16, 1, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, 4, 2), (8, 2, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (16, 4, 3), (12, 3, 1))
    assert_size_stride(primals_23, (16, ), (1, ))
    assert_size_stride(primals_24, (4, 16, 1), (16, 1, 1))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (4, 4, 2), (8, 2, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(reinterpret_tensor(primals_3, (1, 4, 4), (16, 4, 1), 0), primals_1, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (1, 4, 4), (16, 4, 1))
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf33 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf0, primals_2, buf1, buf33, 16, grid=grid(16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(reinterpret_tensor(buf1, (1, 4, 4), (0, 4, 1), 0), primals_4, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf2, (1, 16, 4), (64, 4, 1))
        buf3 = reinterpret_tensor(buf2, (16, 4), (4, 1), 0); del buf2  # reuse
        buf32 = empty_strided_cuda((16, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_1.run(buf3, primals_5, buf32, 64, grid=grid(64), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(reinterpret_tensor(buf3, (1, 16, 4), (0, 4, 1), 0), primals_6, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf4, (1, 4, 4), (16, 4, 1))
        buf5 = reinterpret_tensor(buf0, (4, 4), (4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul, input_6], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_2.run(buf5, primals_2, buf4, primals_7, 16, grid=grid(16), stream=stream0)
        del buf4
        del primals_2
        del primals_7
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(reinterpret_tensor(buf5, (1, 4, 4), (0, 4, 1), 0), primals_8, stride=(1,), padding=(0,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf6, (1, 4, 5), (20, 5, 1))
        buf7 = empty_strided_cuda((4, 5), (5, 1), torch.float32)
        buf31 = empty_strided_cuda((4, 5), (5, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_3.run(buf6, primals_9, buf7, buf31, 20, grid=grid(20), stream=stream0)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(reinterpret_tensor(buf7, (1, 4, 5), (0, 5, 1), 0), primals_10, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf8, (1, 16, 5), (80, 5, 1))
        buf9 = reinterpret_tensor(buf8, (16, 5), (5, 1), 0); del buf8  # reuse
        buf30 = empty_strided_cuda((16, 5), (5, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_4.run(buf9, primals_11, buf30, 80, grid=grid(80), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(reinterpret_tensor(buf9, (1, 16, 5), (0, 5, 1), 0), primals_12, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf10, (1, 4, 5), (20, 5, 1))
        buf11 = reinterpret_tensor(buf6, (4, 5), (5, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [mul_1, input_12], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_5.run(buf11, primals_9, buf10, primals_13, 20, grid=grid(20), stream=stream0)
        del buf10
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(reinterpret_tensor(buf11, (1, 4, 5), (0, 5, 1), 0), primals_14, stride=(1,), padding=(0,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf12, (1, 4, 6), (24, 6, 1))
        buf13 = empty_strided_cuda((4, 6), (6, 1), torch.float32)
        buf29 = empty_strided_cuda((4, 6), (6, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_6.run(buf12, primals_15, buf13, buf29, 24, grid=grid(24), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(reinterpret_tensor(buf13, (1, 4, 6), (0, 6, 1), 0), primals_16, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf14, (1, 16, 6), (96, 6, 1))
        buf15 = reinterpret_tensor(buf14, (16, 6), (6, 1), 0); del buf14  # reuse
        buf28 = empty_strided_cuda((16, 6), (6, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_7.run(buf15, primals_17, buf28, 96, grid=grid(96), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(reinterpret_tensor(buf15, (1, 16, 6), (0, 6, 1), 0), primals_18, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf16, (1, 4, 6), (24, 6, 1))
        buf17 = reinterpret_tensor(buf12, (4, 6), (6, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [mul_2, input_18], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_8.run(buf17, primals_15, buf16, primals_19, 24, grid=grid(24), stream=stream0)
        del buf16
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(reinterpret_tensor(buf17, (1, 4, 6), (0, 6, 1), 0), primals_20, stride=(1,), padding=(0,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf18, (1, 4, 7), (28, 7, 1))
        buf19 = empty_strided_cuda((4, 7), (7, 1), torch.float32)
        buf27 = empty_strided_cuda((4, 7), (7, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9.run(buf18, primals_21, buf19, buf27, 28, grid=grid(28), stream=stream0)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(reinterpret_tensor(buf19, (1, 4, 7), (0, 7, 1), 0), primals_22, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf20, (1, 16, 7), (112, 7, 1))
        buf21 = reinterpret_tensor(buf20, (16, 7), (7, 1), 0); del buf20  # reuse
        buf26 = empty_strided_cuda((16, 7), (7, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_10.run(buf21, primals_23, buf26, 112, grid=grid(112), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(reinterpret_tensor(buf21, (1, 16, 7), (0, 7, 1), 0), primals_24, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf22, (1, 4, 7), (28, 7, 1))
        buf23 = reinterpret_tensor(buf18, (4, 7), (7, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [mul_3, input_24], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_11.run(buf23, primals_21, buf22, primals_25, 28, grid=grid(28), stream=stream0)
        del buf22
        del primals_21
        del primals_25
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(reinterpret_tensor(buf23, (1, 4, 7), (0, 7, 1), 0), primals_26, stride=(1,), padding=(0,), dilation=(1,), transposed=True, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf24, (1, 4, 8), (32, 8, 1))
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_12.run(buf25, primals_27, 32, grid=grid(32), stream=stream0)
        del primals_27
    return (reinterpret_tensor(buf25, (4, 8), (8, 1), 0), primals_1, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, reinterpret_tensor(primals_3, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf1, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf3, (1, 16, 4), (64, 4, 1), 0), reinterpret_tensor(buf5, (1, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf7, (1, 4, 5), (20, 5, 1), 0), reinterpret_tensor(buf9, (1, 16, 5), (80, 5, 1), 0), reinterpret_tensor(buf11, (1, 4, 5), (20, 5, 1), 0), reinterpret_tensor(buf13, (1, 4, 6), (24, 6, 1), 0), reinterpret_tensor(buf15, (1, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf17, (1, 4, 6), (24, 6, 1), 0), reinterpret_tensor(buf19, (1, 4, 7), (28, 7, 1), 0), reinterpret_tensor(buf21, (1, 16, 7), (112, 7, 1), 0), reinterpret_tensor(buf23, (1, 4, 7), (28, 7, 1), 0), buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 2), (8, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 2), (8, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4, 2), (8, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((16, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 4, 2), (8, 2, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
