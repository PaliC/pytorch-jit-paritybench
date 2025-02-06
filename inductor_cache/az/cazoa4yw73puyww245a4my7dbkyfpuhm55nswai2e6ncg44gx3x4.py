# AOT ID: ['6_forward']
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


# kernel path: inductor_cache/mo/cmou2w6ypwfesr66qyio6vebwoievyohbautbah3jcobupm6jnhc.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, YBLOCK])
    tmp3 = tmp0 + tmp2
    tl.store(out_ptr0 + (x2 + 4*y3), tmp3, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/zg/czg3gjf5ebpydz5tvbsbkbhsltfcmlxext3agd7my6zujeoad75n.py
# Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv2d_1 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_5, %primals_6, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_1 = async_compile.triton('triton_poi_fused_convolution_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: inductor_cache/ka/ckaszpmweykiuduuei53cezianbc6ssovcxkrsniucjzv2z5wddz.py
# Topologically Sorted Source Nodes: [truediv, A1_1, mul, A1_2], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   A1_1 => tanh
#   A1_2 => add
#   mul => mul
#   truediv => div
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%bmm, 4), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh, %primals_7), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select, %mul), kwargs = {})
triton_poi_fused_add_div_mul_tanh_2 = async_compile.triton('triton_poi_fused_add_div_mul_tanh_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_tanh_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_tanh_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = 0.25
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.tanh(tmp3)
    tmp7 = tmp4 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dh/cdhi2s74qcfelt3ftqi2imhz7gzabtdbetfjodmv24226m62jfti.py
# Topologically Sorted Source Nodes: [truediv_1, A1_4, mul_1, A1_5], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   A1_4 => tanh_1
#   A1_5 => add_1
#   mul_1 => mul_1
#   truediv_1 => div_1
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%bmm_2, 4), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_1,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_1, %primals_7), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_1, %mul_1), kwargs = {})
triton_poi_fused_add_div_mul_tanh_3 = async_compile.triton('triton_poi_fused_add_div_mul_tanh_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_tanh_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_tanh_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = 0.25
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.tanh(tmp3)
    tmp7 = tmp4 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3g/c3gk3nzccsxvui3c5xnmex7u2zsnel23wk7xo5qkuhwniqzvgwvd.py
# Topologically Sorted Source Nodes: [truediv_2, A1_7, mul_2, A1_8], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   A1_7 => tanh_2
#   A1_8 => add_3
#   mul_2 => mul_2
#   truediv_2 => div_2
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%bmm_4, 4), kwargs = {})
#   %tanh_2 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%div_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tanh_2, %primals_7), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_2, %mul_2), kwargs = {})
triton_poi_fused_add_div_mul_tanh_4 = async_compile.triton('triton_poi_fused_add_div_mul_tanh_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_tanh_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_tanh_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp2 = 0.25
    tmp3 = tmp1 * tmp2
    tmp4 = libdevice.tanh(tmp3)
    tmp7 = tmp4 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fw/cfwbrpllsnikie6ltoubiw3toof6kkimjmegcc7fxhqji6qawnuz.py
# Topologically Sorted Source Nodes: [z, z_1, y, z_2, y_1, y_2, y_3, y_4], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   y => add_2
#   y_1 => add_4
#   y_2 => add_6, mul_4, mul_5, sub
#   y_3 => add_7
#   y_4 => relu
#   z => convolution_2
#   z_1 => convolution_5
#   z_2 => convolution_8
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_9, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %convolution_5 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_19, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_5, %convolution_2), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%view_29, %primals_20, %primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_8, %add_2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %unsqueeze_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_5), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_7), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %primals_1), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), xmask)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 + tmp8
    tmp10 = tmp2 + tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full([1], 0, tl.int32)
    tmp29 = triton_helpers.maximum(tmp28, tmp27)
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr0 + (x3), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lc/clcjraomlyt6peg5t6yliguitbnmmqktxz6vhmc5xjgckezt253l.py
# Topologically Sorted Source Nodes: [se], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   se => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu, [-2]), kwargs = {})
triton_poi_fused_mean_6 = async_compile.triton('triton_poi_fused_mean_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (8 + x0 + 16*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (12 + x0 + 16*x1), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5f/c5f2f2ycxsescrrmns4kl4w7nynmlngvdk22f55xhm2a7wrv6g3a.py
# Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv1d => convolution_9
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_26, %primals_27, [1], [1], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_7 = async_compile.triton('triton_poi_fused_convolution_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: inductor_cache/sy/csyvr2asnny4lioslav2z34iu4m4l5g5y3iyi4apgqddnmxeu43z.py
# Topologically Sorted Source Nodes: [mul_3, y_5], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_3 => mul_6
#   y_5 => add_8
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu, %unsqueeze_8), kwargs = {})
#   %add_8 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %relu), kwargs = {})
triton_poi_fused_add_mul_8 = async_compile.triton('triton_poi_fused_add_mul_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 + tmp0
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zt/czt5o6sqvrrbarrrm643h2c3xypfjgonfqqxtompysdzg6r2bwx4.py
# Topologically Sorted Source Nodes: [se_1], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   se_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%add_8, [-1]), kwargs = {})
triton_poi_fused_mean_9 = async_compile.triton('triton_poi_fused_mean_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dj/cdjmelsqpvdud7535abopzehr24erawpj7y2aeknutuzgqgan6tt.py
# Topologically Sorted Source Nodes: [mul_4, y_6], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_4 => mul_7
#   y_6 => add_9
# Graph fragment:
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_8, %unsqueeze_9), kwargs = {})
#   %add_9 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %add_8), kwargs = {})
triton_poi_fused_add_mul_10 = async_compile.triton('triton_poi_fused_add_mul_10', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x1 = ((xindex // 4) % 4)
    x3 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x1 + 4*x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 + tmp0
    tl.store(out_ptr0 + (x4), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tz/ctzhw2ufum7lpp7zw66iqfiktkaoru35hzifwqbeuly5mdhmdkq6.py
# Topologically Sorted Source Nodes: [mean_2, se_2], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   se_2 => mean_3
# Graph fragment:
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_9, [-1]), kwargs = {})
#   %mean_3 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_2, [-1]), kwargs = {})
triton_poi_fused_mean_11 = async_compile.triton('triton_poi_fused_mean_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 / tmp7
    tmp17 = tmp8 + tmp16
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 / tmp7
    tmp26 = tmp17 + tmp25
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 / tmp7
    tmp35 = tmp26 + tmp34
    tmp36 = tmp35 / tmp7
    tl.store(out_ptr0 + (x0), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i6/ci6elgnoqgyqmrneqwvmsyupvzj4kscpafsnycetouomscsuiuyy.py
# Topologically Sorted Source Nodes: [linear, se1_2], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   linear => add_tensor
#   se1_2 => relu_1
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_31), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
triton_poi_fused_addmm_relu_12 = async_compile.triton('triton_poi_fused_addmm_relu_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 2)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cf/ccfdmyi4onvx3j6xnoyfr2yismez2x5a47rjitodlcwlldz4rhun.py
# Topologically Sorted Source Nodes: [mul_5, y_7], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_5 => mul_8
#   y_7 => add_10
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %unsqueeze_11), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %add_9), kwargs = {})
triton_poi_fused_add_mul_13 = async_compile.triton('triton_poi_fused_add_mul_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3 + tmp0
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_4, (1, ), (1, ))
    assert_size_stride(primals_5, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_6, (1, ), (1, ))
    assert_size_stride(primals_7, (1, ), (1, ))
    assert_size_stride(primals_8, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_11, (1, ), (1, ))
    assert_size_stride(primals_12, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_13, (1, ), (1, ))
    assert_size_stride(primals_14, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_17, (1, ), (1, ))
    assert_size_stride(primals_18, (1, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_19, (1, ), (1, ))
    assert_size_stride(primals_20, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, ), (1, ))
    assert_size_stride(primals_26, (1, 4, 3), (12, 3, 1))
    assert_size_stride(primals_27, (1, ), (1, ))
    assert_size_stride(primals_28, (1, 4, 9), (36, 9, 1))
    assert_size_stride(primals_29, (1, ), (1, ))
    assert_size_stride(primals_30, (2, 4), (4, 1))
    assert_size_stride(primals_31, (2, ), (1, ))
    assert_size_stride(primals_32, (4, 2), (2, 1))
    assert_size_stride(primals_33, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_1, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 1, 4, 4), (16, 16, 4, 1))
        buf2 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf0, primals_4, buf2, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_4
        buf3 = reinterpret_tensor(buf1, (4, 1, 4, 4), (16, 1, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf3, primals_6, 64, grid=grid(64), stream=stream0)
        del primals_6
        buf4 = reinterpret_tensor(buf0, (4, 4, 4), (16, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf3, (4, 4, 4), (16, 4, 1), 0), out=buf4)
        buf5 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv, A1_1, mul, A1_2], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_tanh_2.run(primals_2, buf4, primals_7, buf5, 64, grid=grid(64), stream=stream0)
        buf6 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv, A1_1, mul, A1_2, matmul_1], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_1, (4, 16, 4), (64, 4, 1), 0), buf5, out=buf6)
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(reinterpret_tensor(buf6, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(primals_1, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(primals_1, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 1, 4, 4), (16, 16, 4, 1))
        buf10 = reinterpret_tensor(buf5, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf8, primals_11, buf10, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_11
        buf11 = reinterpret_tensor(buf9, (4, 1, 4, 4), (16, 1, 4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf11, primals_13, 64, grid=grid(64), stream=stream0)
        del primals_13
        buf12 = reinterpret_tensor(buf8, (4, 4, 4), (16, 4, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf11, (4, 4, 4), (16, 4, 1), 0), out=buf12)
        buf13 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_1, A1_4, mul_1, A1_5], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_tanh_3.run(primals_2, buf12, primals_7, buf13, 64, grid=grid(64), stream=stream0)
        buf14 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_1, A1_4, mul_1, A1_5, matmul_3], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_1, (4, 16, 4), (64, 4, 1), 0), buf13, out=buf14)
        # Topologically Sorted Source Nodes: [z_1], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(reinterpret_tensor(buf14, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 4, 4), (64, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(primals_1, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 1, 4, 4), (16, 16, 4, 1))
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(primals_1, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 1, 4, 4), (16, 16, 4, 1))
        buf18 = reinterpret_tensor(buf13, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf16, primals_17, buf18, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_17
        buf19 = reinterpret_tensor(buf17, (4, 1, 4, 4), (16, 1, 4, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_1.run(buf19, primals_19, 64, grid=grid(64), stream=stream0)
        del primals_19
        buf20 = reinterpret_tensor(buf16, (4, 4, 4), (16, 4, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf19, (4, 4, 4), (16, 4, 1), 0), out=buf20)
        buf21 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_2, A1_7, mul_2, A1_8], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_tanh_4.run(primals_2, buf20, primals_7, buf21, 64, grid=grid(64), stream=stream0)
        del primals_2
        buf22 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_2, A1_7, mul_2, A1_8, matmul_5], Original ATen: [aten.div, aten.tanh, aten.mul, aten.add, aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(primals_1, (4, 16, 4), (64, 4, 1), 0), buf21, out=buf22)
        # Topologically Sorted Source Nodes: [z_2], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(reinterpret_tensor(buf22, (4, 4, 4, 4), (64, 16, 4, 1), 0), primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 4, 4, 4), (64, 16, 4, 1))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z, z_1, y, z_2, y_1, y_2, y_3, y_4], Original ATen: [aten.convolution, aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_relu_5.run(buf24, primals_21, buf15, primals_15, buf7, primals_9, primals_22, primals_23, primals_24, primals_25, primals_1, buf25, 256, grid=grid(256), stream=stream0)
        del primals_15
        del primals_21
        del primals_9
        buf26 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [se], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_6.run(buf25, buf26, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_26, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf27, (4, 1, 4), (4, 4, 1))
        buf28 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_7.run(buf28, primals_27, 16, grid=grid(16), stream=stream0)
        del primals_27
        buf29 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [mul_3, y_5], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_8.run(buf29, buf28, 256, grid=grid(256), stream=stream0)
        buf30 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [se_1], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_9.run(buf29, buf30, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d_1], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_28, stride=(1,), padding=(4,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf31, (4, 1, 4), (4, 4, 1))
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [conv1d_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_7.run(buf32, primals_29, 16, grid=grid(16), stream=stream0)
        del primals_29
        buf33 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [mul_4, y_6], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_10.run(buf29, buf32, buf33, 256, grid=grid(256), stream=stream0)
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mean_2, se_2], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_11.run(buf33, buf34, 16, grid=grid(16), stream=stream0)
        buf35 = empty_strided_cuda((4, 2), (2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_30, (4, 2), (1, 4), 0), out=buf35)
        buf36 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [linear, se1_2], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_12.run(buf36, primals_31, 8, grid=grid(8), stream=stream0)
        del primals_31
        buf37 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf36, reinterpret_tensor(primals_32, (2, 4), (1, 2), 0), alpha=1, beta=1, out=buf37)
        del primals_33
        buf38 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [mul_5, y_7], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_13.run(buf33, buf37, buf38, 256, grid=grid(256), stream=stream0)
    return (buf38, primals_1, primals_3, primals_5, primals_7, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_28, buf4, reinterpret_tensor(buf6, (4, 4, 4, 4), (64, 16, 4, 1), 0), buf12, reinterpret_tensor(buf14, (4, 4, 4, 4), (64, 16, 4, 1), 0), buf20, reinterpret_tensor(buf22, (4, 4, 4, 4), (64, 16, 4, 1), 0), buf24, buf26, buf28, buf29, buf30, buf32, buf33, buf34, buf36, buf37, primals_32, primals_30, reinterpret_tensor(buf18, (4, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf19, (4, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf10, (4, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf11, (4, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf2, (4, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf3, (4, 4, 4), (16, 1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((1, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1, 4, 3), (12, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((1, 4, 9), (36, 9, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, 2), (2, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
