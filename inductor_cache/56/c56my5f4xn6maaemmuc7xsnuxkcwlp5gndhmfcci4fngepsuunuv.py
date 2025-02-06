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


# kernel path: inductor_cache/s4/cs4ldyhblxgwadizz3jjaknyygoops7xwlhwdi3omnuhxbt3qaym.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/f3/cf3yvvrx2dp4pn5dwcyzm6qhg7y76yekqxcww4ry23bgzk3jew7k.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/7z/c7znmw7e7kltvytplwctejrbn555lre2qovhefazpqj4lwhtyphs.py
# Topologically Sorted Source Nodes: [x_1, add, relu6, mul, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add => add_2
#   mul => mul_3
#   out => div
#   relu6 => clamp_max, clamp_min
#   x_1 => add_1, mul_1, mul_2, sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 3), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %clamp_max), kwargs = {})
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_3, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, None)
''', device_str='cuda')


# kernel path: inductor_cache/ik/cikz4cwhmtvwp24ttu7hz3akyuxlkdaw226yjcsfttv6pd7zzgrr.py
# Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_3 => add_4, mul_5, mul_6, sub_1
#   x_4 => relu
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %unsqueeze_13), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %unsqueeze_15), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_4,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/gm/cgmjne5dhzuxwyq236cmawf3mfv44gjon3jierwfvlxi6wmxvuw7.py
# Topologically Sorted Source Nodes: [x_9, y], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_9 => add_8, mul_11, mul_12, sub_3
#   y => add_9
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %unsqueeze_29), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %unsqueeze_31), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %add_8), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xq/cxqnpi6fociw6juauhjczfsbkjclopvdzw3yhagdo6h5aqfczbmk.py
# Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_11 => add_11, mul_14, mul_15, sub_4
#   x_12 => relu_2
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %unsqueeze_37), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %unsqueeze_39), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/lh/clhvf3wow3awqgm5nfjkcsn4e3jxwf27hfofwhvimjusdk72b72a.py
# Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_14 => add_13, mul_17, mul_18, sub_5
#   x_15 => relu_3
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_45), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_47), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/s3/cs33zh2bclculd5dla7ijydmxribzqlcs3jxindfcbekk7lhj5pc.py
# Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_17 => add_15, mul_20, mul_21, sub_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_55), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/jb/cjb2jrldu46ojvovtpucqssnhnoc5n5qfgkcodux3y7vhlfkaken.py
# Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_19 => add_17, mul_23, mul_24, sub_7
#   x_20 => relu_4
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_61), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %unsqueeze_63), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/p2/cp2cxnv4xp7cptchb5sd75wy5skhjhkke7lfft3ez4a2we5m5akg.py
# Topologically Sorted Source Nodes: [x_25, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_25 => add_21, mul_29, mul_30, sub_9
#   y_1 => add_22
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_29, %unsqueeze_77), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %unsqueeze_79), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %add_21), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 16*y3), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (y0 + 256*x2 + 4096*y1), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ec/cecip3os55b6cytll5sxf2fqh7t3cuoh2mitmnr5kiv3q46aafsr.py
# Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_26 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_22, %primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_10 = async_compile.triton('triton_poi_fused_convolution_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 256*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 4096*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ck/cck47azc435ajdjl5fttw6do4yzvs6tlxcm7ci5xq6i4y2bzpzcu.py
# Topologically Sorted Source Nodes: [x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_30 => add_26, mul_35, mul_36, sub_11
#   x_31 => relu_7
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_35, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, %unsqueeze_95), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
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


# kernel path: inductor_cache/la/claumhtkvtny7ysnznvwxmjwf35wvlbbnhctwhu7azoqtqbekull.py
# Topologically Sorted Source Nodes: [attn], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   attn => mean
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_7, [-1, -2], True), kwargs = {})
triton_per_fused_mean_12 = async_compile.triton('triton_per_fused_mean_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_12(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 40)
    x1 = xindex // 40
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 40*r2 + 2560*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/to/cto55w7aao3kubtnljh65saihkrn4railkhcl4zeqghjbaueiwko.py
# Topologically Sorted Source Nodes: [attn_1, attn_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   attn_1 => convolution_12
#   attn_2 => relu_8
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_62, %primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_12,), kwargs = {})
triton_poi_fused_convolution_relu_13 = async_compile.triton('triton_poi_fused_convolution_relu_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_13(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 10)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wf/cwf5m733q2wogsrubzjwrnqykk7ubvgnnv6ykl7qfz6i7dpddun6.py
# Topologically Sorted Source Nodes: [attn_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   attn_3 => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_8, %primals_64, %primals_65, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_14 = async_compile.triton('triton_poi_fused_convolution_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5n/c5ng4czpp7fsf7k5nxc3v47izkivrczdrzu26fhgsvgsvr7rtn2z.py
# Topologically Sorted Source Nodes: [mul_1, x_32, neg, result, neg_1, result_1, y_2], Original ATen: [aten.mul, aten.add, aten.neg, aten.threshold]
# Source node to ATen node mapping:
#   mul_1 => mul_37
#   neg => neg
#   neg_1 => neg_1
#   result => full_default, le, where
#   result_1 => full_default_1, le_1, where_1
#   x_32 => add_27
#   y_2 => mul_38
# Graph fragment:
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_13, 0.2), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_37, 0.5), kwargs = {})
#   %neg : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_27,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg, -1), kwargs = {})
#   %full_default : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le, %full_default, %neg), kwargs = {})
#   %neg_1 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where,), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_1, 0), kwargs = {})
#   %full_default_1 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_1, %full_default_1, %neg_1), kwargs = {})
#   %mul_38 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_7, %where_1), kwargs = {})
triton_poi_fused_add_mul_neg_threshold_15 = async_compile.triton('triton_poi_fused_add_mul_neg_threshold_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_neg_threshold_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_neg_threshold_15(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 40)
    x2 = xindex // 2560
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 40*x2), xmask, eviction_policy='evict_last')
    tmp2 = 0.2
    tmp3 = tmp1 * tmp2
    tmp4 = 0.5
    tmp5 = tmp3 + tmp4
    tmp6 = -tmp5
    tmp7 = -1.0
    tmp8 = tmp6 <= tmp7
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tmp10 = -tmp9
    tmp11 = 0.0
    tmp12 = tmp10 <= tmp11
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tmp0 * tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6j/c6jbklwls3khel7oss35clrxyd7dk7fu6pjipy2jjhscfr3qwjw7.py
# Topologically Sorted Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_34 => add_29, mul_40, mul_41, sub_12
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_97), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_101), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_103), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qf/cqfdlzpo2in2ulbundlajjg2vtoqruz6iyr6a4f5orozzb6z3tvj.py
# Topologically Sorted Source Nodes: [x_36, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_36 => add_31, mul_43, mul_44, sub_13
#   x_37 => relu_9
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_105), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_109), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_111), kwargs = {})
#   %relu_9 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/zl/czlpdcv5xbjqfprd2wmi4n67tee3sup3qd5dn6sbde6yfubgr7wz.py
# Topologically Sorted Source Nodes: [attn_4], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   attn_4 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_10, [-1, -2], True), kwargs = {})
triton_per_fused_mean_18 = async_compile.triton('triton_per_fused_mean_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_18(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 4096*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvxgq65s6m7iqsmwnquvk3uifgiw3bkuadw65o42kx2c4u6dcxm.py
# Topologically Sorted Source Nodes: [attn_5, attn_6], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   attn_5 => convolution_17
#   attn_6 => relu_11
# Graph fragment:
#   %convolution_17 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_1, %primals_81, %primals_82, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_17,), kwargs = {})
triton_poi_fused_convolution_relu_19 = async_compile.triton('triton_poi_fused_convolution_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gc/cgceosvhdx72uah5fi5urf72dxps7xfba6bsmspookaotyu4sbz4.py
# Topologically Sorted Source Nodes: [result, result_1, attn_7, mul_3, x_41, neg_2, result_2, neg_3, result_3], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
# Source node to ATen node mapping:
#   attn_7 => convolution_18
#   mul_3 => mul_48
#   neg_2 => neg_2
#   neg_3 => neg_3
#   result => full_default
#   result_1 => full_default_1
#   result_2 => le_2, where_2
#   result_3 => le_3, where_3
#   x_41 => add_34
# Graph fragment:
#   %full_default : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_83, %primals_84, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_18, 0.2), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, 0.5), kwargs = {})
#   %neg_2 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_34,), kwargs = {})
#   %le_2 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_2, -1), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_2, %full_default, %neg_2), kwargs = {})
#   %neg_3 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where_2,), kwargs = {})
#   %le_3 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_3, 0), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_3, %full_default_1, %neg_3), kwargs = {})
triton_poi_fused_add_convolution_mul_neg_threshold_20 = async_compile.triton('triton_poi_fused_add_convolution_mul_neg_threshold_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_neg_threshold_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_neg_threshold_20(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp5 = 0.5
    tmp6 = tmp4 + tmp5
    tmp7 = -tmp6
    tmp8 = -1.0
    tmp9 = tmp7 <= tmp8
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tmp11 = -tmp10
    tmp12 = 0.0
    tmp13 = tmp11 <= tmp12
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
    tl.store(in_out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6y/c6yqrpgt5aknw76sl3zgyshb6jftixjwoqx67s2bkvhla553fpp3.py
# Topologically Sorted Source Nodes: [y_3], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   y_3 => mul_49
# Graph fragment:
#   %mul_49 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%relu_10, %where_3), kwargs = {})
triton_poi_fused_mul_21 = async_compile.triton('triton_poi_fused_mul_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/sr/csrtd6kp4fz2orcm45nfqdv2tnqesyqclxrrfsgfzrhzossilvji.py
# Topologically Sorted Source Nodes: [x_43, y_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_43 => add_36, mul_51, mul_52, sub_15
#   y_4 => add_37
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_121), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_125), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %unsqueeze_127), kwargs = {})
#   %add_37 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %add_36), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ly/clyobz5huld5kf7fnh77b7naj4mwpynk6m3cifkkmvkwlqk3b5jx.py
# Topologically Sorted Source Nodes: [x_52, y_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_52 => add_44, mul_62, mul_63, sub_18
#   y_6 => add_45
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_145), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_62, %unsqueeze_149), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %unsqueeze_151), kwargs = {})
#   %add_45 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %add_44), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 24
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    tmp0 = tl.load(in_ptr0 + (x2 + 24*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 24*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (y0 + 64*x2 + 1536*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ul/culsq65jbckwkltashw275kvy4ekruu6xqejinittjbh2d6x7sqa.py
# Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_53 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_45, %primals_109, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_24 = async_compile.triton('triton_poi_fused_convolution_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 1536*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/wm/cwmluv3mz4bygbcae4zezsiv3glx6l3kdznq4b4epnkl6td7fkwd.py
# Topologically Sorted Source Nodes: [x_54, add_8, relu6_1, mul_7, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_8 => add_48
#   mul_7 => mul_67
#   out_1 => div_1
#   relu6_1 => clamp_max_1, clamp_min_1
#   x_54 => add_47, mul_65, mul_66, sub_19
# Graph fragment:
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_153), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %unsqueeze_157), kwargs = {})
#   %add_47 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %unsqueeze_159), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_47, 3), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_48, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_47, %clamp_max_1), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_67, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oq/coq4dx6p3qyts6g4vo67vmylkskmuq3a66dvb6yiqcjxlsuckx6e.py
# Topologically Sorted Source Nodes: [x_56, add_9, relu6_2, mul_8, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_9 => add_51
#   mul_8 => mul_71
#   out_2 => div_2
#   relu6_2 => clamp_max_2, clamp_min_2
#   x_56 => add_50, mul_69, mul_70, sub_20
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_161), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %unsqueeze_165), kwargs = {})
#   %add_50 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %unsqueeze_167), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_50, 3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_51, 0), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_50, %clamp_max_2), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_71, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ty/ctyh2i3zju7fjcicxroqcfk25odvhlf5zrmabnbzal6bmceqgthj.py
# Topologically Sorted Source Nodes: [x_58], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_58 => add_53, mul_73, mul_74, sub_21
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_169), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_173), kwargs = {})
#   %add_53 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_175), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w5/cw5uxyek7bybd7qhjqi3tnr2dfu53uw7u44kxqwepj36a5xkgffo.py
# Topologically Sorted Source Nodes: [x_60, add_10, relu6_3, mul_9, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_10 => add_56
#   mul_9 => mul_78
#   out_3 => div_3
#   relu6_3 => clamp_max_3, clamp_min_3
#   x_60 => add_55, mul_76, mul_77, sub_22
# Graph fragment:
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_177), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_181), kwargs = {})
#   %add_55 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_183), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, 3), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_56, 0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6), kwargs = {})
#   %mul_78 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_55, %clamp_max_3), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_78, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_28', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6656
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 104)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t7/ct7wykacw3obhu67jmdhfwjhrhbd62yxw3mv3swzwalyirgokvml.py
# Topologically Sorted Source Nodes: [x_64, y_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_64 => add_61, mul_84, mul_85, sub_24
#   y_7 => add_62
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_193), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_84, %unsqueeze_197), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_85, %unsqueeze_199), kwargs = {})
#   %add_62 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_53, %add_61), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_29', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 40)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckad65ueiqikvt643s2522mkunrni37ovkdhgnd2hedemxydeosi.py
# Topologically Sorted Source Nodes: [x_66, add_13, relu6_5, mul_11, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_13 => add_65
#   mul_11 => mul_89
#   out_5 => div_5
#   relu6_5 => clamp_max_5, clamp_min_5
#   x_66 => add_64, mul_87, mul_88, sub_25
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_201), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_87, %unsqueeze_205), kwargs = {})
#   %add_64 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_88, %unsqueeze_207), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_64, 3), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_65, 0), kwargs = {})
#   %clamp_max_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 6), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_64, %clamp_max_5), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_89, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 96)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t5/ct5gduhp5xjsm4jqil4xitb5tbf34fbf5vfgcbpmm5pgt5uexr5k.py
# Topologically Sorted Source Nodes: [x_78, add_19, relu6_9, mul_15, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_19 => add_83
#   mul_15 => mul_111
#   out_9 => div_9
#   relu6_9 => clamp_max_9, clamp_min_9
#   x_78 => add_82, mul_109, mul_110, sub_31
# Graph fragment:
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_37, %unsqueeze_249), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_253), kwargs = {})
#   %add_82 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_255), kwargs = {})
#   %add_83 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_82, 3), kwargs = {})
#   %clamp_min_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_83, 0), kwargs = {})
#   %clamp_max_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_9, 6), kwargs = {})
#   %mul_111 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_82, %clamp_max_9), kwargs = {})
#   %div_9 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_111, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ce/cce5pv7qwwbd4lbu26mrskdxyj72vjeeomwefx5oeoawrgeyjvpu.py
# Topologically Sorted Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_80 => add_85, mul_113, mul_114, sub_32
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_257), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_261), kwargs = {})
#   %add_85 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_263), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_32', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hg/chgkil2ouv5s54venbmegfulfhi42v7d6dmwjxryehlc2ukhna3z.py
# Topologically Sorted Source Nodes: [add_20, relu6_10, mul_16, out_10, attn_12], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
# Source node to ATen node mapping:
#   add_20 => add_86
#   attn_12 => mean_3
#   mul_16 => mul_115
#   out_10 => div_10
#   relu6_10 => clamp_max_10, clamp_min_10
# Graph fragment:
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_85, 3), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_86, 0), kwargs = {})
#   %clamp_max_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %clamp_max_10), kwargs = {})
#   %div_10 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_115, 6), kwargs = {})
#   %mean_3 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_10, [-1, -2], True), kwargs = {})
triton_per_fused_add_div_hardtanh_mean_mul_33 = async_compile.triton('triton_per_fused_add_div_hardtanh_mean_mul_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_mean_mul_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_hardtanh_mean_mul_33(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 240)
    x1 = xindex // 240
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 240*r2 + 3840*x1), xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 16.0
    tmp15 = tmp13 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rw/crwm36dunqt3aqqakksa42nbfxu6jgfkeygg34vzgjabqqaltxg5.py
# Topologically Sorted Source Nodes: [attn_13, attn_14], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   attn_13 => convolution_39
#   attn_14 => relu_15
# Graph fragment:
#   %convolution_39 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_3, %primals_179, %primals_180, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_39,), kwargs = {})
triton_poi_fused_convolution_relu_34 = async_compile.triton('triton_poi_fused_convolution_relu_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_34(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 60)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7o/c7okddgivw3glxwy5pgenm4glye3fj7qbomvgrfw3a4u5mcgpll3.py
# Topologically Sorted Source Nodes: [result, result_1, attn_15, mul_17, x_81, neg_6, result_6, neg_7, result_7], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
# Source node to ATen node mapping:
#   attn_15 => convolution_40
#   mul_17 => mul_116
#   neg_6 => neg_6
#   neg_7 => neg_7
#   result => full_default
#   result_1 => full_default_1
#   result_6 => le_6, where_6
#   result_7 => le_7, where_7
#   x_81 => add_87
# Graph fragment:
#   %full_default : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_40 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_15, %primals_181, %primals_182, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_40, 0.2), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, 0.5), kwargs = {})
#   %neg_6 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_87,), kwargs = {})
#   %le_6 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_6, -1), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_6, %full_default, %neg_6), kwargs = {})
#   %neg_7 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where_6,), kwargs = {})
#   %le_7 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_7, 0), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_7, %full_default_1, %neg_7), kwargs = {})
triton_poi_fused_add_convolution_mul_neg_threshold_35 = async_compile.triton('triton_poi_fused_add_convolution_mul_neg_threshold_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_neg_threshold_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_neg_threshold_35(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 240)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp5 = 0.5
    tmp6 = tmp4 + tmp5
    tmp7 = -tmp6
    tmp8 = -1.0
    tmp9 = tmp7 <= tmp8
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tmp11 = -tmp10
    tmp12 = 0.0
    tmp13 = tmp11 <= tmp12
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
    tl.store(in_out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5f/c5f4bpiq5u7plyjiawz47pfjrkle66cjhsmta3vszddxjbgqcvn5.py
# Topologically Sorted Source Nodes: [add_20, relu6_10, mul_16, out_10, y_10], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_20 => add_86
#   mul_16 => mul_115
#   out_10 => div_10
#   relu6_10 => clamp_max_10, clamp_min_10
#   y_10 => mul_117
# Graph fragment:
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_85, 3), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_86, 0), kwargs = {})
#   %clamp_max_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, %clamp_max_10), kwargs = {})
#   %div_10 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_115, 6), kwargs = {})
#   %mul_117 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_10, %where_7), kwargs = {})
triton_poi_fused_add_div_hardtanh_mul_36 = async_compile.triton('triton_poi_fused_add_div_hardtanh_mul_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_hardtanh_mul_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_hardtanh_mul_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 240)
    x2 = xindex // 3840
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 240*x2), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/db/cdbwbkjvjnkvr2b34ujrsqolwqucbwvdclvsdyajb4gvlsh4fzt7.py
# Topologically Sorted Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_83 => add_89, mul_119, mul_120, sub_33
# Graph fragment:
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_265), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %unsqueeze_267), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %unsqueeze_269), kwargs = {})
#   %add_89 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %unsqueeze_271), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 56)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ns/cnsf3dg5xrvjzacnhahfozki3s3imk4jjcmbg76plqfzzqi7egiu.py
# Topologically Sorted Source Nodes: [x_85, add_22, relu6_11, mul_19, out_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_22 => add_92
#   mul_19 => mul_124
#   out_11 => div_11
#   relu6_11 => clamp_max_11, clamp_min_11
#   x_85 => add_91, mul_122, mul_123, sub_34
# Graph fragment:
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_273), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_122, %unsqueeze_277), kwargs = {})
#   %add_91 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_123, %unsqueeze_279), kwargs = {})
#   %add_92 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_91, 3), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_92, 0), kwargs = {})
#   %clamp_max_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 6), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_91, %clamp_max_11), kwargs = {})
#   %div_11 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_124, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 336)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x6/cx6qmqnm4clwqla466g4htn36ryl3oebrn3q7ftdpfojpbx4xipp.py
# Topologically Sorted Source Nodes: [x_87], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_87 => add_94, mul_126, mul_127, sub_35
# Graph fragment:
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_281), kwargs = {})
#   %mul_126 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_127 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_126, %unsqueeze_285), kwargs = {})
#   %add_94 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_127, %unsqueeze_287), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_39', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 336)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2nxc5ndytjv7cthaif2n7kytbnjoptmshuscyhfvcg76pcaayz.py
# Topologically Sorted Source Nodes: [add_23, relu6_12, mul_20, out_12, attn_16], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
# Source node to ATen node mapping:
#   add_23 => add_95
#   attn_16 => mean_4
#   mul_20 => mul_128
#   out_12 => div_12
#   relu6_12 => clamp_max_12, clamp_min_12
# Graph fragment:
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, 3), kwargs = {})
#   %clamp_min_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_95, 0), kwargs = {})
#   %clamp_max_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, %clamp_max_12), kwargs = {})
#   %div_12 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_128, 6), kwargs = {})
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_12, [-1, -2], True), kwargs = {})
triton_per_fused_add_div_hardtanh_mean_mul_40 = async_compile.triton('triton_per_fused_add_div_hardtanh_mean_mul_40', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_hardtanh_mean_mul_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_hardtanh_mean_mul_40(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 336)
    x1 = xindex // 336
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 336*r2 + 5376*x1), xmask, other=0.0)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(xmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp14 = 16.0
    tmp15 = tmp13 / tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/b5/cb5cx2y3q7n4yn64vv5dcjd2egyf4cc5uf6ooqmjdh65bknhqtk3.py
# Topologically Sorted Source Nodes: [attn_17, attn_18], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   attn_17 => convolution_44
#   attn_18 => relu_16
# Graph fragment:
#   %convolution_44 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_4, %primals_198, %primals_199, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_44,), kwargs = {})
triton_poi_fused_convolution_relu_41 = async_compile.triton('triton_poi_fused_convolution_relu_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_41(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 84)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pm/cpm3kesdgbpwtvsqomiwgzityqv3he6j7jjyvppucom67jr4x2eb.py
# Topologically Sorted Source Nodes: [result, result_1, attn_19, mul_21, x_88, neg_8, result_8, neg_9, result_9], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
# Source node to ATen node mapping:
#   attn_19 => convolution_45
#   mul_21 => mul_129
#   neg_8 => neg_8
#   neg_9 => neg_9
#   result => full_default
#   result_1 => full_default_1
#   result_8 => le_8, where_8
#   result_9 => le_9, where_9
#   x_88 => add_96
# Graph fragment:
#   %full_default : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_45 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_16, %primals_200, %primals_201, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_45, 0.2), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_129, 0.5), kwargs = {})
#   %neg_8 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_96,), kwargs = {})
#   %le_8 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_8, -1), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_8, %full_default, %neg_8), kwargs = {})
#   %neg_9 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where_8,), kwargs = {})
#   %le_9 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_9, 0), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_9, %full_default_1, %neg_9), kwargs = {})
triton_poi_fused_add_convolution_mul_neg_threshold_42 = async_compile.triton('triton_poi_fused_add_convolution_mul_neg_threshold_42', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_neg_threshold_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_neg_threshold_42(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 336)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp5 = 0.5
    tmp6 = tmp4 + tmp5
    tmp7 = -tmp6
    tmp8 = -1.0
    tmp9 = tmp7 <= tmp8
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tmp11 = -tmp10
    tmp12 = 0.0
    tmp13 = tmp11 <= tmp12
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
    tl.store(in_out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nb/cnbvkd65we65zf7vi5egk4cjpe4wyv2nil2zdhiemwywcyxlmqlx.py
# Topologically Sorted Source Nodes: [add_23, relu6_12, mul_20, out_12, y_11], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_23 => add_95
#   mul_20 => mul_128
#   out_12 => div_12
#   relu6_12 => clamp_max_12, clamp_min_12
#   y_11 => mul_130
# Graph fragment:
#   %add_95 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, 3), kwargs = {})
#   %clamp_min_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_95, 0), kwargs = {})
#   %clamp_max_12 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_12, 6), kwargs = {})
#   %mul_128 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, %clamp_max_12), kwargs = {})
#   %div_12 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_128, 6), kwargs = {})
#   %mul_130 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_12, %where_9), kwargs = {})
triton_poi_fused_add_div_hardtanh_mul_43 = async_compile.triton('triton_poi_fused_add_div_hardtanh_mul_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_hardtanh_mul_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_hardtanh_mul_43(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 336)
    x2 = xindex // 5376
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 336*x2), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/px/cpxb4v643lz7ysrn7tzmaftxpeg754g2tonxpo6r3koqnwy4cb7t.py
# Topologically Sorted Source Nodes: [x_90, y_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_90 => add_98, mul_132, mul_133, sub_36
#   y_12 => add_99
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_289), kwargs = {})
#   %mul_132 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_132, %unsqueeze_293), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_133, %unsqueeze_295), kwargs = {})
#   %add_99 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_89, %add_98), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_44', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 56
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 56*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 56*y3), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1, 1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (y0 + 16*x2 + 896*y1), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/g6/cg6cxzrbu2ys4mxuh7hlcetclehs2iqwgcdtevjixpt43oh7h3ji.py
# Topologically Sorted Source Nodes: [x_91], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_91 => convolution_47
# Graph fragment:
#   %convolution_47 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_99, %primals_207, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_45 = async_compile.triton('triton_poi_fused_convolution_45', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_45(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 224
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 56)
    y1 = yindex // 56
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 56*x2 + 896*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chwe36txcrinef2ohzo3oxzsngi2lrfnhwvhgw6gnbgw32c3q7sd.py
# Topologically Sorted Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_94 => add_104, mul_139, mul_140, sub_38
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %unsqueeze_305), kwargs = {})
#   %mul_139 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_139, %unsqueeze_309), kwargs = {})
#   %add_104 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_140, %unsqueeze_311), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 336)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bs/cbsues2s7jfsgrasvw5e2natbawjjproifucjctiqt5ngknda4ex.py
# Topologically Sorted Source Nodes: [add_27, relu6_14, mul_24, out_14, attn_20], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
# Source node to ATen node mapping:
#   add_27 => add_105
#   attn_20 => mean_5
#   mul_24 => mul_141
#   out_14 => div_14
#   relu6_14 => clamp_max_14, clamp_min_14
# Graph fragment:
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, 3), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_105, 0), kwargs = {})
#   %clamp_max_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 6), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %clamp_max_14), kwargs = {})
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_141, 6), kwargs = {})
#   %mean_5 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_14, [-1, -2], True), kwargs = {})
triton_poi_fused_add_div_hardtanh_mean_mul_47 = async_compile.triton('triton_poi_fused_add_div_hardtanh_mean_mul_47', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_hardtanh_mean_mul_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_hardtanh_mean_mul_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 336)
    x1 = xindex // 336
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1344*x1), xmask)
    tmp10 = tl.load(in_ptr0 + (336 + x0 + 1344*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (672 + x0 + 1344*x1), xmask)
    tmp24 = tl.load(in_ptr0 + (1008 + x0 + 1344*x1), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp11, tmp3)
    tmp13 = triton_helpers.minimum(tmp12, tmp5)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 * tmp8
    tmp16 = tmp9 + tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = triton_helpers.minimum(tmp19, tmp5)
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp16 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = triton_helpers.maximum(tmp25, tmp3)
    tmp27 = triton_helpers.minimum(tmp26, tmp5)
    tmp28 = tmp24 * tmp27
    tmp29 = tmp28 * tmp8
    tmp30 = tmp23 + tmp29
    tmp31 = 4.0
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ti/ctiux6i6nk2gyg4nloeyusu4mjagvp7jxnwjy6ourfmvozsgnzp4.py
# Topologically Sorted Source Nodes: [add_27, relu6_14, mul_24, out_14, y_13], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_27 => add_105
#   mul_24 => mul_141
#   out_14 => div_14
#   relu6_14 => clamp_max_14, clamp_min_14
#   y_13 => mul_143
# Graph fragment:
#   %add_105 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, 3), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_105, 0), kwargs = {})
#   %clamp_max_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 6), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_104, %clamp_max_14), kwargs = {})
#   %div_14 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_141, 6), kwargs = {})
#   %mul_143 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_14, %where_11), kwargs = {})
triton_poi_fused_add_div_hardtanh_mul_48 = async_compile.triton('triton_poi_fused_add_div_hardtanh_mul_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_hardtanh_mul_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_hardtanh_mul_48(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 336)
    x2 = xindex // 1344
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 336*x2), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kp/ckplyje25i5gjacdsqsizdfz2xj6q5idylsvmxyo2ymgkrrj4ujh.py
# Topologically Sorted Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_97 => add_108, mul_145, mul_146, sub_39
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_313), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_145, %unsqueeze_317), kwargs = {})
#   %add_108 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_146, %unsqueeze_319), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_49', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uq/cuqel2ygqo24t5lbe74chfdaux5e46krwshf26ceposedrtpf5jr.py
# Topologically Sorted Source Nodes: [x_99, add_29, relu6_15, mul_27, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_29 => add_111
#   mul_27 => mul_150
#   out_15 => div_15
#   relu6_15 => clamp_max_15, clamp_min_15
#   x_99 => add_110, mul_148, mul_149, sub_40
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_321), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_325), kwargs = {})
#   %add_110 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_327), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_110, 3), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_111, 0), kwargs = {})
#   %clamp_max_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_110, %clamp_max_15), kwargs = {})
#   %div_15 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_150, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 480)
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
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(in_out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cj/ccjypft2ijsifziifij4s4lsj25lc77k22rsamadayrt55vevana.py
# Topologically Sorted Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_101 => add_113, mul_152, mul_153, sub_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_329), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_152, %unsqueeze_333), kwargs = {})
#   %add_113 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_153, %unsqueeze_335), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 480)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yl/cyl254cm2ttsn2roxseij727pwnc7x326mbtwuy3p3ezowpo2oad.py
# Topologically Sorted Source Nodes: [add_30, relu6_16, mul_28, out_16, attn_24], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
# Source node to ATen node mapping:
#   add_30 => add_114
#   attn_24 => mean_6
#   mul_28 => mul_154
#   out_16 => div_16
#   relu6_16 => clamp_max_16, clamp_min_16
# Graph fragment:
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, 3), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_114, 0), kwargs = {})
#   %clamp_max_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_113, %clamp_max_16), kwargs = {})
#   %div_16 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_154, 6), kwargs = {})
#   %mean_6 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%div_16, [-1, -2], True), kwargs = {})
triton_poi_fused_add_div_hardtanh_mean_mul_52 = async_compile.triton('triton_poi_fused_add_div_hardtanh_mean_mul_52', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_hardtanh_mean_mul_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_hardtanh_mean_mul_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 480)
    x1 = xindex // 480
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1920*x1), xmask)
    tmp10 = tl.load(in_ptr0 + (480 + x0 + 1920*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (960 + x0 + 1920*x1), xmask)
    tmp24 = tl.load(in_ptr0 + (1440 + x0 + 1920*x1), xmask)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp10 + tmp1
    tmp12 = triton_helpers.maximum(tmp11, tmp3)
    tmp13 = triton_helpers.minimum(tmp12, tmp5)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 * tmp8
    tmp16 = tmp9 + tmp15
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp18, tmp3)
    tmp20 = triton_helpers.minimum(tmp19, tmp5)
    tmp21 = tmp17 * tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp16 + tmp22
    tmp25 = tmp24 + tmp1
    tmp26 = triton_helpers.maximum(tmp25, tmp3)
    tmp27 = triton_helpers.minimum(tmp26, tmp5)
    tmp28 = tmp24 * tmp27
    tmp29 = tmp28 * tmp8
    tmp30 = tmp23 + tmp29
    tmp31 = 4.0
    tmp32 = tmp30 / tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mm/cmmihuhzlr4lkaaekql2cm6xuk2ewvf3xh5hihtb6s6fzaae5gcc.py
# Topologically Sorted Source Nodes: [attn_25, attn_26], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   attn_25 => convolution_54
#   attn_26 => relu_18
# Graph fragment:
#   %convolution_54 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_6, %primals_236, %primals_237, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_54,), kwargs = {})
triton_poi_fused_convolution_relu_53 = async_compile.triton('triton_poi_fused_convolution_relu_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_53(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 120)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bm/cbm2q3z75asi2zfr2n463nvq2dxsaym2r4yq6idweai7ow2vdcxy.py
# Topologically Sorted Source Nodes: [result, result_1, attn_27, mul_29, x_102, neg_12, result_12, neg_13, result_13], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
# Source node to ATen node mapping:
#   attn_27 => convolution_55
#   mul_29 => mul_155
#   neg_12 => neg_12
#   neg_13 => neg_13
#   result => full_default
#   result_1 => full_default_1
#   result_12 => le_12, where_12
#   result_13 => le_13, where_13
#   x_102 => add_115
# Graph fragment:
#   %full_default : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], -1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=8] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convolution_55 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_18, %primals_238, %primals_239, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_55, 0.2), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, 0.5), kwargs = {})
#   %neg_12 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%add_115,), kwargs = {})
#   %le_12 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_12, -1), kwargs = {})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%le_12, %full_default, %neg_12), kwargs = {})
#   %neg_13 : [num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%where_12,), kwargs = {})
#   %le_13 : [num_users=2] = call_function[target=torch.ops.aten.le.Scalar](args = (%neg_13, 0), kwargs = {})
#   %where_13 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%le_13, %full_default_1, %neg_13), kwargs = {})
triton_poi_fused_add_convolution_mul_neg_threshold_54 = async_compile.triton('triton_poi_fused_add_convolution_mul_neg_threshold_54', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_neg_threshold_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_mul_neg_threshold_54(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 480)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.2
    tmp4 = tmp2 * tmp3
    tmp5 = 0.5
    tmp6 = tmp4 + tmp5
    tmp7 = -tmp6
    tmp8 = -1.0
    tmp9 = tmp7 <= tmp8
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tmp11 = -tmp10
    tmp12 = 0.0
    tmp13 = tmp11 <= tmp12
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
    tl.store(out_ptr1 + (x2), tmp13, xmask)
    tl.store(in_out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7w/c7wgaojsvxhmjeq7bwubnwd7qmh7a2m6tw5fznvnu4m2e37vfebm.py
# Topologically Sorted Source Nodes: [add_30, relu6_16, mul_28, out_16, y_14], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_30 => add_114
#   mul_28 => mul_154
#   out_16 => div_16
#   relu6_16 => clamp_max_16, clamp_min_16
#   y_14 => mul_156
# Graph fragment:
#   %add_114 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_113, 3), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_114, 0), kwargs = {})
#   %clamp_max_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_113, %clamp_max_16), kwargs = {})
#   %div_16 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_154, 6), kwargs = {})
#   %mul_156 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_16, %where_13), kwargs = {})
triton_poi_fused_add_div_hardtanh_mul_55 = async_compile.triton('triton_poi_fused_add_div_hardtanh_mul_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_hardtanh_mul_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_hardtanh_mul_55(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 480)
    x2 = xindex // 1920
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp10 = tl.load(in_ptr0 + (x0 + 480*x2), xmask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = 0.16666666666666666
    tmp9 = tmp7 * tmp8
    tmp11 = tmp9 * tmp10
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5a/c5asdh5ijuat37ohh7jhw3w2asqtthfvk2rqlbhgmqy2ptqh6anc.py
# Topologically Sorted Source Nodes: [x_104, y_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   x_104 => add_117, mul_158, mul_159, sub_42
#   y_15 => add_118
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_56, %unsqueeze_337), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %unsqueeze_341), kwargs = {})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %unsqueeze_343), kwargs = {})
#   %add_118 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %add_117), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_56', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 80)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sp/cspfsutu7xstcnxfgihwgmg2vjiaeyt7xe4dxhhqq5srtcao2b3s.py
# Topologically Sorted Source Nodes: [x_113, add_37, relu6_19, mul_35, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
# Source node to ATen node mapping:
#   add_37 => add_131
#   mul_35 => mul_176
#   out_19 => div_19
#   relu6_19 => clamp_max_19, clamp_min_19
#   x_113 => add_130, mul_174, mul_175, sub_46
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_369), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_174, %unsqueeze_373), kwargs = {})
#   %add_130 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_175, %unsqueeze_375), kwargs = {})
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_130, 3), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_131, 0), kwargs = {})
#   %clamp_max_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 6), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_130, %clamp_max_19), kwargs = {})
#   %div_19 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_176, 6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_57', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 4)
    y3 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x1 + 480*y0), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 3.0
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = tmp15 * tmp21
    tmp23 = 0.16666666666666666
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr1 + (y2 + 4*x1 + 1920*y3), tmp24, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268 = args
    args.clear()
    assert_size_stride(primals_1, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (8, ), (1, ))
    assert_size_stride(primals_4, (8, ), (1, ))
    assert_size_stride(primals_5, (8, ), (1, ))
    assert_size_stride(primals_6, (8, ), (1, ))
    assert_size_stride(primals_7, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_8, (8, ), (1, ))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, ), (1, ))
    assert_size_stride(primals_12, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (8, ), (1, ))
    assert_size_stride(primals_15, (8, ), (1, ))
    assert_size_stride(primals_16, (8, ), (1, ))
    assert_size_stride(primals_17, (8, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_18, (8, ), (1, ))
    assert_size_stride(primals_19, (8, ), (1, ))
    assert_size_stride(primals_20, (8, ), (1, ))
    assert_size_stride(primals_21, (8, ), (1, ))
    assert_size_stride(primals_22, (32, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_27, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_28, (32, ), (1, ))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (32, ), (1, ))
    assert_size_stride(primals_32, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_33, (16, ), (1, ))
    assert_size_stride(primals_34, (16, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (40, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_38, (40, ), (1, ))
    assert_size_stride(primals_39, (40, ), (1, ))
    assert_size_stride(primals_40, (40, ), (1, ))
    assert_size_stride(primals_41, (40, ), (1, ))
    assert_size_stride(primals_42, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_43, (40, ), (1, ))
    assert_size_stride(primals_44, (40, ), (1, ))
    assert_size_stride(primals_45, (40, ), (1, ))
    assert_size_stride(primals_46, (40, ), (1, ))
    assert_size_stride(primals_47, (16, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_48, (16, ), (1, ))
    assert_size_stride(primals_49, (16, ), (1, ))
    assert_size_stride(primals_50, (16, ), (1, ))
    assert_size_stride(primals_51, (16, ), (1, ))
    assert_size_stride(primals_52, (40, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_53, (40, ), (1, ))
    assert_size_stride(primals_54, (40, ), (1, ))
    assert_size_stride(primals_55, (40, ), (1, ))
    assert_size_stride(primals_56, (40, ), (1, ))
    assert_size_stride(primals_57, (40, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_58, (40, ), (1, ))
    assert_size_stride(primals_59, (40, ), (1, ))
    assert_size_stride(primals_60, (40, ), (1, ))
    assert_size_stride(primals_61, (40, ), (1, ))
    assert_size_stride(primals_62, (10, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_63, (10, ), (1, ))
    assert_size_stride(primals_64, (40, 10, 1, 1), (10, 1, 1, 1))
    assert_size_stride(primals_65, (40, ), (1, ))
    assert_size_stride(primals_66, (24, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_67, (24, ), (1, ))
    assert_size_stride(primals_68, (24, ), (1, ))
    assert_size_stride(primals_69, (24, ), (1, ))
    assert_size_stride(primals_70, (24, ), (1, ))
    assert_size_stride(primals_71, (64, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, ), (1, ))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_81, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_82, (16, ), (1, ))
    assert_size_stride(primals_83, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_85, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_86, (24, ), (1, ))
    assert_size_stride(primals_87, (24, ), (1, ))
    assert_size_stride(primals_88, (24, ), (1, ))
    assert_size_stride(primals_89, (24, ), (1, ))
    assert_size_stride(primals_90, (64, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (64, ), (1, ))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_95, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_96, (64, ), (1, ))
    assert_size_stride(primals_97, (64, ), (1, ))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_101, (16, ), (1, ))
    assert_size_stride(primals_102, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_105, (24, ), (1, ))
    assert_size_stride(primals_106, (24, ), (1, ))
    assert_size_stride(primals_107, (24, ), (1, ))
    assert_size_stride(primals_108, (24, ), (1, ))
    assert_size_stride(primals_109, (120, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_110, (120, ), (1, ))
    assert_size_stride(primals_111, (120, ), (1, ))
    assert_size_stride(primals_112, (120, ), (1, ))
    assert_size_stride(primals_113, (120, ), (1, ))
    assert_size_stride(primals_114, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_115, (120, ), (1, ))
    assert_size_stride(primals_116, (120, ), (1, ))
    assert_size_stride(primals_117, (120, ), (1, ))
    assert_size_stride(primals_118, (120, ), (1, ))
    assert_size_stride(primals_119, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_120, (40, ), (1, ))
    assert_size_stride(primals_121, (40, ), (1, ))
    assert_size_stride(primals_122, (40, ), (1, ))
    assert_size_stride(primals_123, (40, ), (1, ))
    assert_size_stride(primals_124, (104, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_125, (104, ), (1, ))
    assert_size_stride(primals_126, (104, ), (1, ))
    assert_size_stride(primals_127, (104, ), (1, ))
    assert_size_stride(primals_128, (104, ), (1, ))
    assert_size_stride(primals_129, (104, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (104, ), (1, ))
    assert_size_stride(primals_131, (104, ), (1, ))
    assert_size_stride(primals_132, (104, ), (1, ))
    assert_size_stride(primals_133, (104, ), (1, ))
    assert_size_stride(primals_134, (40, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(primals_135, (40, ), (1, ))
    assert_size_stride(primals_136, (40, ), (1, ))
    assert_size_stride(primals_137, (40, ), (1, ))
    assert_size_stride(primals_138, (40, ), (1, ))
    assert_size_stride(primals_139, (96, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_140, (96, ), (1, ))
    assert_size_stride(primals_141, (96, ), (1, ))
    assert_size_stride(primals_142, (96, ), (1, ))
    assert_size_stride(primals_143, (96, ), (1, ))
    assert_size_stride(primals_144, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (96, ), (1, ))
    assert_size_stride(primals_146, (96, ), (1, ))
    assert_size_stride(primals_147, (96, ), (1, ))
    assert_size_stride(primals_148, (96, ), (1, ))
    assert_size_stride(primals_149, (40, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_150, (40, ), (1, ))
    assert_size_stride(primals_151, (40, ), (1, ))
    assert_size_stride(primals_152, (40, ), (1, ))
    assert_size_stride(primals_153, (40, ), (1, ))
    assert_size_stride(primals_154, (96, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_155, (96, ), (1, ))
    assert_size_stride(primals_156, (96, ), (1, ))
    assert_size_stride(primals_157, (96, ), (1, ))
    assert_size_stride(primals_158, (96, ), (1, ))
    assert_size_stride(primals_159, (96, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_160, (96, ), (1, ))
    assert_size_stride(primals_161, (96, ), (1, ))
    assert_size_stride(primals_162, (96, ), (1, ))
    assert_size_stride(primals_163, (96, ), (1, ))
    assert_size_stride(primals_164, (40, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_165, (40, ), (1, ))
    assert_size_stride(primals_166, (40, ), (1, ))
    assert_size_stride(primals_167, (40, ), (1, ))
    assert_size_stride(primals_168, (40, ), (1, ))
    assert_size_stride(primals_169, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_170, (240, ), (1, ))
    assert_size_stride(primals_171, (240, ), (1, ))
    assert_size_stride(primals_172, (240, ), (1, ))
    assert_size_stride(primals_173, (240, ), (1, ))
    assert_size_stride(primals_174, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_175, (240, ), (1, ))
    assert_size_stride(primals_176, (240, ), (1, ))
    assert_size_stride(primals_177, (240, ), (1, ))
    assert_size_stride(primals_178, (240, ), (1, ))
    assert_size_stride(primals_179, (60, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_180, (60, ), (1, ))
    assert_size_stride(primals_181, (240, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(primals_182, (240, ), (1, ))
    assert_size_stride(primals_183, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_184, (56, ), (1, ))
    assert_size_stride(primals_185, (56, ), (1, ))
    assert_size_stride(primals_186, (56, ), (1, ))
    assert_size_stride(primals_187, (56, ), (1, ))
    assert_size_stride(primals_188, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_189, (336, ), (1, ))
    assert_size_stride(primals_190, (336, ), (1, ))
    assert_size_stride(primals_191, (336, ), (1, ))
    assert_size_stride(primals_192, (336, ), (1, ))
    assert_size_stride(primals_193, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_194, (336, ), (1, ))
    assert_size_stride(primals_195, (336, ), (1, ))
    assert_size_stride(primals_196, (336, ), (1, ))
    assert_size_stride(primals_197, (336, ), (1, ))
    assert_size_stride(primals_198, (84, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_199, (84, ), (1, ))
    assert_size_stride(primals_200, (336, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(primals_201, (336, ), (1, ))
    assert_size_stride(primals_202, (56, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_203, (56, ), (1, ))
    assert_size_stride(primals_204, (56, ), (1, ))
    assert_size_stride(primals_205, (56, ), (1, ))
    assert_size_stride(primals_206, (56, ), (1, ))
    assert_size_stride(primals_207, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_208, (336, ), (1, ))
    assert_size_stride(primals_209, (336, ), (1, ))
    assert_size_stride(primals_210, (336, ), (1, ))
    assert_size_stride(primals_211, (336, ), (1, ))
    assert_size_stride(primals_212, (336, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_213, (336, ), (1, ))
    assert_size_stride(primals_214, (336, ), (1, ))
    assert_size_stride(primals_215, (336, ), (1, ))
    assert_size_stride(primals_216, (336, ), (1, ))
    assert_size_stride(primals_217, (84, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_218, (84, ), (1, ))
    assert_size_stride(primals_219, (336, 84, 1, 1), (84, 1, 1, 1))
    assert_size_stride(primals_220, (336, ), (1, ))
    assert_size_stride(primals_221, (80, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_222, (80, ), (1, ))
    assert_size_stride(primals_223, (80, ), (1, ))
    assert_size_stride(primals_224, (80, ), (1, ))
    assert_size_stride(primals_225, (80, ), (1, ))
    assert_size_stride(primals_226, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_227, (480, ), (1, ))
    assert_size_stride(primals_228, (480, ), (1, ))
    assert_size_stride(primals_229, (480, ), (1, ))
    assert_size_stride(primals_230, (480, ), (1, ))
    assert_size_stride(primals_231, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_232, (480, ), (1, ))
    assert_size_stride(primals_233, (480, ), (1, ))
    assert_size_stride(primals_234, (480, ), (1, ))
    assert_size_stride(primals_235, (480, ), (1, ))
    assert_size_stride(primals_236, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_237, (120, ), (1, ))
    assert_size_stride(primals_238, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_239, (480, ), (1, ))
    assert_size_stride(primals_240, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_241, (80, ), (1, ))
    assert_size_stride(primals_242, (80, ), (1, ))
    assert_size_stride(primals_243, (80, ), (1, ))
    assert_size_stride(primals_244, (80, ), (1, ))
    assert_size_stride(primals_245, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_246, (480, ), (1, ))
    assert_size_stride(primals_247, (480, ), (1, ))
    assert_size_stride(primals_248, (480, ), (1, ))
    assert_size_stride(primals_249, (480, ), (1, ))
    assert_size_stride(primals_250, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_251, (480, ), (1, ))
    assert_size_stride(primals_252, (480, ), (1, ))
    assert_size_stride(primals_253, (480, ), (1, ))
    assert_size_stride(primals_254, (480, ), (1, ))
    assert_size_stride(primals_255, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_256, (120, ), (1, ))
    assert_size_stride(primals_257, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_258, (480, ), (1, ))
    assert_size_stride(primals_259, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_260, (80, ), (1, ))
    assert_size_stride(primals_261, (80, ), (1, ))
    assert_size_stride(primals_262, (80, ), (1, ))
    assert_size_stride(primals_263, (80, ), (1, ))
    assert_size_stride(primals_264, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_265, (480, ), (1, ))
    assert_size_stride(primals_266, (480, ), (1, ))
    assert_size_stride(primals_267, (480, ), (1, ))
    assert_size_stride(primals_268, (480, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((8, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 24, 9, grid=grid(24, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 8, 32, 32), (8192, 1, 256, 8))
        buf3 = empty_strided_cuda((4, 8, 32, 32), (8192, 1, 256, 8), torch.float32)
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [x_1, add, relu6, mul, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_2.run(buf4, buf2, primals_3, primals_4, primals_5, primals_6, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 8, 32, 32), (8192, 1, 256, 8))
        buf6 = empty_strided_cuda((4, 8, 32, 32), (8192, 1, 256, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf5, primals_8, primals_9, primals_10, primals_11, buf6, 32768, grid=grid(32768), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf7, (4, 8, 32, 32), (8192, 1, 256, 8))
        buf8 = empty_strided_cuda((4, 8, 32, 32), (8192, 1, 256, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_6, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf7, primals_13, primals_14, primals_15, primals_16, buf8, 32768, grid=grid(32768), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 8, 32, 32), (8192, 1, 256, 8))
        buf10 = empty_strided_cuda((4, 8, 32, 32), (8192, 1, 256, 8), torch.float32)
        # Topologically Sorted Source Nodes: [x_9, y], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf4, buf9, primals_18, primals_19, primals_20, primals_21, buf10, 32768, grid=grid(32768), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf12 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_11, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf11, primals_23, primals_24, primals_25, primals_26, buf12, 131072, grid=grid(131072), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_27, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf13, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf14 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf13, primals_28, primals_29, primals_30, primals_31, buf14, 32768, grid=grid(32768), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf16 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf15, primals_33, primals_34, primals_35, primals_36, buf16, 16384, grid=grid(16384), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 40, 16, 16), (10240, 1, 640, 40))
        buf18 = empty_strided_cuda((4, 40, 16, 16), (10240, 1, 640, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf17, primals_38, primals_39, primals_40, primals_41, buf18, 40960, grid=grid(40960), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf19, (4, 40, 16, 16), (10240, 1, 640, 40))
        buf20 = empty_strided_cuda((4, 40, 16, 16), (10240, 1, 640, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_22, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf19, primals_43, primals_44, primals_45, primals_46, buf20, 40960, grid=grid(40960), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 16, 16, 16), (4096, 1, 256, 16))
        buf22 = empty_strided_cuda((4, 16, 16, 16), (4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_25, y_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf16, buf21, primals_48, primals_49, primals_50, primals_51, buf22, 1024, 16, grid=grid(1024, 16), stream=stream0)
        del primals_51
        buf23 = empty_strided_cuda((4, 16, 16, 16), (4096, 1, 256, 16), torch.float32)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_10.run(buf22, buf23, 64, 256, grid=grid(64, 256), stream=stream0)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 40, 16, 16), (10240, 1, 640, 40))
        buf25 = empty_strided_cuda((4, 40, 16, 16), (10240, 1, 640, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf24, primals_53, primals_54, primals_55, primals_56, buf25, 40960, grid=grid(40960), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_57, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
        assert_size_stride(buf26, (4, 40, 8, 8), (2560, 1, 320, 40))
        buf27 = empty_strided_cuda((4, 40, 8, 8), (2560, 1, 320, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf26, primals_58, primals_59, primals_60, primals_61, buf27, 10240, grid=grid(10240), stream=stream0)
        buf28 = empty_strided_cuda((4, 40, 1, 1), (40, 1, 160, 160), torch.float32)
        buf29 = reinterpret_tensor(buf28, (4, 40, 1, 1), (40, 1, 40, 40), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [attn], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_12.run(buf29, buf27, 160, 64, grid=grid(160), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 10, 1, 1), (10, 1, 10, 10))
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [attn_1, attn_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_13.run(buf31, primals_63, 40, grid=grid(40), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [attn_3], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 40, 1, 1), (40, 1, 40, 40))
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [attn_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_14.run(buf33, primals_65, 160, grid=grid(160), stream=stream0)
        del primals_65
        buf34 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [mul_1, x_32, neg, result, neg_1, result_1, y_2], Original ATen: [aten.mul, aten.add, aten.neg, aten.threshold]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_neg_threshold_15.run(buf34, buf33, 10240, grid=grid(10240), stream=stream0)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 24, 8, 8), (1536, 1, 192, 24))
        buf36 = empty_strided_cuda((4, 24, 8, 8), (1536, 1, 192, 24), torch.float32)
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf35, primals_67, primals_68, primals_69, primals_70, buf36, 6144, grid=grid(6144), stream=stream0)
        del primals_70
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_71, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf38 = reinterpret_tensor(buf23, (4, 64, 8, 8), (4096, 1, 512, 64), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_36, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf37, primals_72, primals_73, primals_74, primals_75, buf38, 16384, grid=grid(16384), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_76, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf39, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf40 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_39, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf39, primals_77, primals_78, primals_79, primals_80, buf40, 16384, grid=grid(16384), stream=stream0)
        buf41 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf42 = reinterpret_tensor(buf41, (4, 64, 1, 1), (64, 1, 64, 64), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [attn_4], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_18.run(buf42, buf40, 256, 64, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_5], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 16, 1, 1), (16, 1, 16, 16))
        buf44 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [attn_5, attn_6], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_19.run(buf44, primals_82, 64, grid=grid(64), stream=stream0)
        del primals_82
        # Topologically Sorted Source Nodes: [attn_7], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_83, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 64, 1, 1), (64, 1, 64, 64))
        buf46 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        buf47 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        buf48 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [result, result_1, attn_7, mul_3, x_41, neg_2, result_2, neg_3, result_3], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_neg_threshold_20.run(buf48, primals_84, buf46, buf47, 256, grid=grid(256), stream=stream0)
        del primals_84
        buf49 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [y_3], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_21.run(buf49, buf48, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 24, 8, 8), (1536, 1, 192, 24))
        buf51 = empty_strided_cuda((4, 24, 8, 8), (1536, 1, 192, 24), torch.float32)
        # Topologically Sorted Source Nodes: [x_43, y_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_22.run(buf36, buf50, primals_86, primals_87, primals_88, primals_89, buf51, 6144, grid=grid(6144), stream=stream0)
        del primals_89
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf53 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_45, x_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf52, primals_91, primals_92, primals_93, primals_94, buf53, 16384, grid=grid(16384), stream=stream0)
        del primals_94
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_95, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf54, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf55 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_48, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf54, primals_96, primals_97, primals_98, primals_99, buf55, 16384, grid=grid(16384), stream=stream0)
        buf56 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf57 = reinterpret_tensor(buf56, (4, 64, 1, 1), (64, 1, 64, 64), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [attn_8], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_18.run(buf57, buf55, 256, 64, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_9], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 16, 1, 1), (16, 1, 16, 16))
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [attn_9, attn_10], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_19.run(buf59, primals_101, 64, grid=grid(64), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [attn_11], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 64, 1, 1), (64, 1, 64, 64))
        buf61 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        buf62 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.bool)
        buf63 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [result, result_1, attn_11, mul_5, x_50, neg_4, result_4, neg_5, result_5], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_neg_threshold_20.run(buf63, primals_103, buf61, buf62, 256, grid=grid(256), stream=stream0)
        del primals_103
        buf64 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [y_5], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_21.run(buf64, buf63, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_104, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 24, 8, 8), (1536, 1, 192, 24))
        buf66 = empty_strided_cuda((4, 24, 8, 8), (1536, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_52, y_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf51, buf65, primals_105, primals_106, primals_107, primals_108, buf66, 256, 24, grid=grid(256, 24), stream=stream0)
        del primals_108
        buf67 = empty_strided_cuda((4, 24, 8, 8), (1536, 1, 192, 24), torch.float32)
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_24.run(buf66, buf67, 96, 64, grid=grid(96, 64), stream=stream0)
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 120, 8, 8), (7680, 1, 960, 120))
        buf69 = empty_strided_cuda((4, 120, 8, 8), (7680, 1, 960, 120), torch.float32)
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_54, add_8, relu6_1, mul_7, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_25.run(buf70, buf68, primals_110, primals_111, primals_112, primals_113, 30720, grid=grid(30720), stream=stream0)
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_114, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf71, (4, 120, 4, 4), (1920, 1, 480, 120))
        buf72 = empty_strided_cuda((4, 120, 4, 4), (1920, 1, 480, 120), torch.float32)
        buf73 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [x_56, add_9, relu6_2, mul_8, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_26.run(buf73, buf71, primals_115, primals_116, primals_117, primals_118, 7680, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 40, 4, 4), (640, 1, 160, 40))
        buf75 = empty_strided_cuda((4, 40, 4, 4), (640, 1, 160, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_27.run(buf74, primals_120, primals_121, primals_122, primals_123, buf75, 2560, grid=grid(2560), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf77 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_60, add_10, relu6_3, mul_9, out_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_28.run(buf78, buf76, primals_125, primals_126, primals_127, primals_128, 6656, grid=grid(6656), stream=stream0)
        # Topologically Sorted Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=104, bias=None)
        assert_size_stride(buf79, (4, 104, 4, 4), (1664, 1, 416, 104))
        buf80 = empty_strided_cuda((4, 104, 4, 4), (1664, 1, 416, 104), torch.float32)
        buf81 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [x_62, add_11, relu6_4, mul_10, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_28.run(buf81, buf79, primals_130, primals_131, primals_132, primals_133, 6656, grid=grid(6656), stream=stream0)
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 40, 4, 4), (640, 1, 160, 40))
        buf83 = empty_strided_cuda((4, 40, 4, 4), (640, 1, 160, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_64, y_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf75, buf82, primals_135, primals_136, primals_137, primals_138, buf83, 2560, grid=grid(2560), stream=stream0)
        del primals_138
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf85 = reinterpret_tensor(buf67, (4, 96, 4, 4), (1536, 1, 384, 96), 0); del buf67  # reuse
        buf86 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_66, add_13, relu6_5, mul_11, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30.run(buf86, buf84, primals_140, primals_141, primals_142, primals_143, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf87, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf88 = empty_strided_cuda((4, 96, 4, 4), (1536, 1, 384, 96), torch.float32)
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_68, add_14, relu6_6, mul_12, out_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30.run(buf89, buf87, primals_145, primals_146, primals_147, primals_148, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 40, 4, 4), (640, 1, 160, 40))
        buf91 = empty_strided_cuda((4, 40, 4, 4), (640, 1, 160, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_70, y_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf83, buf90, primals_150, primals_151, primals_152, primals_153, buf91, 2560, grid=grid(2560), stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf93 = empty_strided_cuda((4, 96, 4, 4), (1536, 1, 384, 96), torch.float32)
        buf94 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [x_72, add_16, relu6_7, mul_13, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30.run(buf94, buf92, primals_155, primals_156, primals_157, primals_158, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=96, bias=None)
        assert_size_stride(buf95, (4, 96, 4, 4), (1536, 1, 384, 96))
        buf96 = empty_strided_cuda((4, 96, 4, 4), (1536, 1, 384, 96), torch.float32)
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [x_74, add_17, relu6_8, mul_14, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_30.run(buf97, buf95, primals_160, primals_161, primals_162, primals_163, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 40, 4, 4), (640, 1, 160, 40))
        buf99 = empty_strided_cuda((4, 40, 4, 4), (640, 1, 160, 40), torch.float32)
        # Topologically Sorted Source Nodes: [x_76, y_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf91, buf98, primals_165, primals_166, primals_167, primals_168, buf99, 2560, grid=grid(2560), stream=stream0)
        del primals_168
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 240, 4, 4), (3840, 1, 960, 240))
        buf101 = empty_strided_cuda((4, 240, 4, 4), (3840, 1, 960, 240), torch.float32)
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [x_78, add_19, relu6_9, mul_15, out_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_31.run(buf102, buf100, primals_170, primals_171, primals_172, primals_173, 15360, grid=grid(15360), stream=stream0)
        # Topologically Sorted Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf103, (4, 240, 4, 4), (3840, 1, 960, 240))
        buf104 = empty_strided_cuda((4, 240, 4, 4), (3840, 1, 960, 240), torch.float32)
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_32.run(buf103, primals_175, primals_176, primals_177, primals_178, buf104, 15360, grid=grid(15360), stream=stream0)
        buf105 = empty_strided_cuda((4, 240, 1, 1), (240, 1, 960, 960), torch.float32)
        buf106 = reinterpret_tensor(buf105, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [add_20, relu6_10, mul_16, out_10, attn_12], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_hardtanh_mean_mul_33.run(buf106, buf104, 960, 16, grid=grid(960), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_13], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 60, 1, 1), (60, 1, 60, 60))
        buf108 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [attn_13, attn_14], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_34.run(buf108, primals_180, 240, grid=grid(240), stream=stream0)
        del primals_180
        # Topologically Sorted Source Nodes: [attn_15], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 240, 1, 1), (240, 1, 240, 240))
        buf110 = empty_strided_cuda((4, 240, 1, 1), (240, 1, 240, 240), torch.bool)
        buf111 = empty_strided_cuda((4, 240, 1, 1), (240, 1, 240, 240), torch.bool)
        buf112 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [result, result_1, attn_15, mul_17, x_81, neg_6, result_6, neg_7, result_7], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_neg_threshold_35.run(buf112, primals_182, buf110, buf111, 960, grid=grid(960), stream=stream0)
        del primals_182
        buf113 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [add_20, relu6_10, mul_16, out_10, y_10], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mul_36.run(buf113, buf112, 15360, grid=grid(15360), stream=stream0)
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 56, 4, 4), (896, 1, 224, 56))
        buf115 = empty_strided_cuda((4, 56, 4, 4), (896, 1, 224, 56), torch.float32)
        # Topologically Sorted Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf114, primals_184, primals_185, primals_186, primals_187, buf115, 3584, grid=grid(3584), stream=stream0)
        del primals_187
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 336, 4, 4), (5376, 1, 1344, 336))
        buf117 = empty_strided_cuda((4, 336, 4, 4), (5376, 1, 1344, 336), torch.float32)
        buf118 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_85, add_22, relu6_11, mul_19, out_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_38.run(buf118, buf116, primals_189, primals_190, primals_191, primals_192, 21504, grid=grid(21504), stream=stream0)
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_193, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf119, (4, 336, 4, 4), (5376, 1, 1344, 336))
        buf120 = empty_strided_cuda((4, 336, 4, 4), (5376, 1, 1344, 336), torch.float32)
        # Topologically Sorted Source Nodes: [x_87], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_39.run(buf119, primals_194, primals_195, primals_196, primals_197, buf120, 21504, grid=grid(21504), stream=stream0)
        buf121 = empty_strided_cuda((4, 336, 1, 1), (336, 1, 1344, 1344), torch.float32)
        buf122 = reinterpret_tensor(buf121, (4, 336, 1, 1), (336, 1, 336, 336), 0); del buf121  # reuse
        # Topologically Sorted Source Nodes: [add_23, relu6_12, mul_20, out_12, attn_16], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_hardtanh_mean_mul_40.run(buf122, buf120, 1344, 16, grid=grid(1344), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_17], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 84, 1, 1), (84, 1, 84, 84))
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [attn_17, attn_18], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_41.run(buf124, primals_199, 336, grid=grid(336), stream=stream0)
        del primals_199
        # Topologically Sorted Source Nodes: [attn_19], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 336, 1, 1), (336, 1, 336, 336))
        buf126 = empty_strided_cuda((4, 336, 1, 1), (336, 1, 336, 336), torch.bool)
        buf127 = empty_strided_cuda((4, 336, 1, 1), (336, 1, 336, 336), torch.bool)
        buf128 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [result, result_1, attn_19, mul_21, x_88, neg_8, result_8, neg_9, result_9], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_neg_threshold_42.run(buf128, primals_201, buf126, buf127, 1344, grid=grid(1344), stream=stream0)
        del primals_201
        buf129 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [add_23, relu6_12, mul_20, out_12, y_11], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mul_43.run(buf129, buf128, 21504, grid=grid(21504), stream=stream0)
        # Topologically Sorted Source Nodes: [x_89], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 56, 4, 4), (896, 1, 224, 56))
        buf131 = empty_strided_cuda((4, 56, 4, 4), (896, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_90, y_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_44.run(buf115, buf130, primals_203, primals_204, primals_205, primals_206, buf131, 64, 56, grid=grid(64, 56), stream=stream0)
        del primals_206
        buf132 = empty_strided_cuda((4, 56, 4, 4), (896, 1, 224, 56), torch.float32)
        # Topologically Sorted Source Nodes: [x_91], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_45.run(buf131, buf132, 224, 16, grid=grid(224, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [x_91], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 336, 4, 4), (5376, 1, 1344, 336))
        del buf132
        buf134 = empty_strided_cuda((4, 336, 4, 4), (5376, 1, 1344, 336), torch.float32)
        buf135 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [x_92, add_26, relu6_13, mul_23, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_38.run(buf135, buf133, primals_208, primals_209, primals_210, primals_211, 21504, grid=grid(21504), stream=stream0)
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_212, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf136, (4, 336, 2, 2), (1344, 1, 672, 336))
        buf137 = empty_strided_cuda((4, 336, 2, 2), (1344, 1, 672, 336), torch.float32)
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf136, primals_213, primals_214, primals_215, primals_216, buf137, 5376, grid=grid(5376), stream=stream0)
        buf138 = empty_strided_cuda((4, 336, 1, 1), (336, 1, 336, 336), torch.float32)
        # Topologically Sorted Source Nodes: [add_27, relu6_14, mul_24, out_14, attn_20], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mean_mul_47.run(buf137, buf138, 1344, grid=grid(1344), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_21], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 84, 1, 1), (84, 1, 84, 84))
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [attn_21, attn_22], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_41.run(buf140, primals_218, 336, grid=grid(336), stream=stream0)
        del primals_218
        # Topologically Sorted Source Nodes: [attn_23], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 336, 1, 1), (336, 1, 336, 336))
        buf142 = empty_strided_cuda((4, 336, 1, 1), (336, 1, 336, 336), torch.bool)
        buf143 = empty_strided_cuda((4, 336, 1, 1), (336, 1, 336, 336), torch.bool)
        buf144 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [result, result_1, attn_23, mul_25, x_95, neg_10, result_10, neg_11, result_11], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_neg_threshold_42.run(buf144, primals_220, buf142, buf143, 1344, grid=grid(1344), stream=stream0)
        del primals_220
        buf145 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [add_27, relu6_14, mul_24, out_14, y_13], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mul_48.run(buf145, buf144, 5376, grid=grid(5376), stream=stream0)
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 80, 2, 2), (320, 1, 160, 80))
        buf147 = empty_strided_cuda((4, 80, 2, 2), (320, 1, 160, 80), torch.float32)
        # Topologically Sorted Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_49.run(buf146, primals_222, primals_223, primals_224, primals_225, buf147, 1280, grid=grid(1280), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 480, 2, 2), (1920, 1, 960, 480))
        buf149 = empty_strided_cuda((4, 480, 2, 2), (1920, 1, 960, 480), torch.float32)
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [x_99, add_29, relu6_15, mul_27, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_50.run(buf150, buf148, primals_227, primals_228, primals_229, primals_230, 7680, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_231, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf151, (4, 480, 2, 2), (1920, 1, 960, 480))
        buf152 = empty_strided_cuda((4, 480, 2, 2), (1920, 1, 960, 480), torch.float32)
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf151, primals_232, primals_233, primals_234, primals_235, buf152, 7680, grid=grid(7680), stream=stream0)
        buf153 = empty_strided_cuda((4, 480, 1, 1), (480, 1, 480, 480), torch.float32)
        # Topologically Sorted Source Nodes: [add_30, relu6_16, mul_28, out_16, attn_24], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mean_mul_52.run(buf152, buf153, 1920, grid=grid(1920), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_25], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 120, 1, 1), (120, 1, 120, 120))
        buf155 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [attn_25, attn_26], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_53.run(buf155, primals_237, 480, grid=grid(480), stream=stream0)
        del primals_237
        # Topologically Sorted Source Nodes: [attn_27], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 480, 1, 1), (480, 1, 480, 480))
        buf157 = empty_strided_cuda((4, 480, 1, 1), (480, 1, 480, 480), torch.bool)
        buf158 = empty_strided_cuda((4, 480, 1, 1), (480, 1, 480, 480), torch.bool)
        buf159 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [result, result_1, attn_27, mul_29, x_102, neg_12, result_12, neg_13, result_13], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_neg_threshold_54.run(buf159, primals_239, buf157, buf158, 1920, grid=grid(1920), stream=stream0)
        del primals_239
        buf160 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [add_30, relu6_16, mul_28, out_16, y_14], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mul_55.run(buf160, buf159, 7680, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 80, 2, 2), (320, 1, 160, 80))
        buf162 = empty_strided_cuda((4, 80, 2, 2), (320, 1, 160, 80), torch.float32)
        # Topologically Sorted Source Nodes: [x_104, y_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_56.run(buf147, buf161, primals_241, primals_242, primals_243, primals_244, buf162, 1280, grid=grid(1280), stream=stream0)
        del primals_244
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_245, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 480, 2, 2), (1920, 1, 960, 480))
        buf164 = empty_strided_cuda((4, 480, 2, 2), (1920, 1, 960, 480), torch.float32)
        buf165 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [x_106, add_33, relu6_17, mul_31, out_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_50.run(buf165, buf163, primals_246, primals_247, primals_248, primals_249, 7680, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_250, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf166, (4, 480, 2, 2), (1920, 1, 960, 480))
        buf167 = empty_strided_cuda((4, 480, 2, 2), (1920, 1, 960, 480), torch.float32)
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf166, primals_251, primals_252, primals_253, primals_254, buf167, 7680, grid=grid(7680), stream=stream0)
        buf168 = empty_strided_cuda((4, 480, 1, 1), (480, 1, 480, 480), torch.float32)
        # Topologically Sorted Source Nodes: [add_34, relu6_18, mul_32, out_18, attn_28], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mean_mul_52.run(buf167, buf168, 1920, grid=grid(1920), stream=stream0)
        # Topologically Sorted Source Nodes: [attn_29], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_255, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 120, 1, 1), (120, 1, 120, 120))
        buf170 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [attn_29, attn_30], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_53.run(buf170, primals_256, 480, grid=grid(480), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [attn_31], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_257, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 480, 1, 1), (480, 1, 480, 480))
        buf172 = empty_strided_cuda((4, 480, 1, 1), (480, 1, 480, 480), torch.bool)
        buf173 = empty_strided_cuda((4, 480, 1, 1), (480, 1, 480, 480), torch.bool)
        buf174 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [result, result_1, attn_31, mul_33, x_109, neg_14, result_14, neg_15, result_15], Original ATen: [aten.threshold, aten.convolution, aten.mul, aten.add, aten.neg]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_mul_neg_threshold_54.run(buf174, primals_258, buf172, buf173, 1920, grid=grid(1920), stream=stream0)
        del primals_258
        buf175 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [add_34, relu6_18, mul_32, out_18, y_16], Original ATen: [aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_hardtanh_mul_55.run(buf175, buf174, 7680, grid=grid(7680), stream=stream0)
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 80, 2, 2), (320, 1, 160, 80))
        buf177 = empty_strided_cuda((4, 80, 2, 2), (320, 1, 160, 80), torch.float32)
        # Topologically Sorted Source Nodes: [x_111, y_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_56.run(buf162, buf176, primals_260, primals_261, primals_262, primals_263, buf177, 1280, grid=grid(1280), stream=stream0)
        del primals_263
        # Topologically Sorted Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 480, 2, 2), (1920, 1, 960, 480))
        buf180 = empty_strided_cuda((4, 480, 2, 2), (1920, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_113, add_37, relu6_19, mul_35, out_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.hardtanh, aten.mul, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_div_hardtanh_mul_57.run(buf178, primals_265, primals_266, primals_267, primals_268, buf180, 16, 480, grid=grid(16, 480), stream=stream0)
    return (buf22, buf66, buf131, buf180, buf0, buf1, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_64, primals_66, primals_67, primals_68, primals_69, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_83, primals_85, primals_86, primals_87, primals_88, primals_90, primals_91, primals_92, primals_93, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_104, primals_105, primals_106, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_181, primals_183, primals_184, primals_185, primals_186, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_219, primals_221, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_243, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_257, primals_259, primals_260, primals_261, primals_262, primals_264, primals_265, primals_266, primals_267, primals_268, buf2, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf24, buf25, buf26, buf29, buf31, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf42, buf44, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf57, buf59, buf61, buf62, buf63, buf64, buf65, buf66, buf68, buf70, buf71, buf73, buf74, buf75, buf76, buf78, buf79, buf81, buf82, buf83, buf84, buf86, buf87, buf89, buf90, buf91, buf92, buf94, buf95, buf97, buf98, buf99, buf100, buf102, buf103, buf106, buf108, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf118, buf119, buf122, buf124, buf126, buf127, buf128, buf129, buf130, buf131, buf133, buf135, buf136, buf138, buf140, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf150, buf151, buf153, buf155, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf165, buf166, buf168, buf170, buf172, buf173, buf174, buf175, buf176, buf177, buf178, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((8, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((8, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((40, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((16, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((40, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((40, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((10, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((40, 10, 1, 1), (10, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((24, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((120, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((104, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((104, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((104, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((40, 104, 1, 1), (104, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((96, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((40, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((96, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((96, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((40, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((60, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((60, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((240, 60, 1, 1), (60, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((84, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((336, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((56, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((336, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((84, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((84, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((336, 84, 1, 1), (84, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((80, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
