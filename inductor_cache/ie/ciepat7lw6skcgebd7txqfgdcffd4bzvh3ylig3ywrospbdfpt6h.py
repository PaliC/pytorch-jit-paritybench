# AOT ID: ['5_forward']
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


# kernel path: inductor_cache/qe/cqebnspl3i5c5aa6hecw4u3xphsyonyv55drbwiineob4zow5os2.py
# Topologically Sorted Source Nodes: [rep_weight], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   rep_weight => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_4, %permute, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x1), xmask & ymask)
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/3e/c3ed64vnur6pagki2ulxcs7qmhrrg45lhrtw23nporjulzxorqqx.py
# Topologically Sorted Source Nodes: [ones, rep_bias], Original ATen: [aten.ones, aten.mul]
# Source node to ATen node mapping:
#   ones => full_default
#   rep_bias => mul
# Graph fragment:
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([1, 4, 3, 3], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%full_default, %view), kwargs = {})
triton_poi_fused_mul_ones_1 = async_compile.triton('triton_poi_fused_mul_ones_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_ones_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_ones_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 9
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp1 * tmp0
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ab/cabd6plkryzf7fvodwsxx4ixfe7syxxzurxsljkbm2qvg2xokrx3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %select_23, 0, 2), kwargs = {})
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 9
    x0 = (xindex % 9)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (2))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp5 = tl.load(in_ptr1 + (18 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.load(in_ptr1 + (9 + x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (0))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp6 = tmp4 * tmp5
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp1 == tmp7
    tmp9 = tmp0 == tmp7
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = tmp7 == tmp14
    tmp16 = tmp0 == tmp14
    tmp20 = tmp18 * tmp19
    tmp21 = 0.0
    tmp22 = tl.where(tmp16, tmp20, tmp21)
    tmp23 = tl.where(tmp15, tmp22, tmp21)
    tmp24 = tl.where(tmp9, tmp13, tmp23)
    tmp25 = tmp1 == tmp14
    tmp26 = tl.where(tmp25, tmp22, tmp21)
    tmp27 = tl.where(tmp8, tmp24, tmp26)
    tmp28 = tl.where(tmp2, tmp6, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/37/c37dmpguh2547wt57ufbjt6p2i3vidzcpknoco4t7ejeuuvq73xe.py
# Topologically Sorted Source Nodes: [k1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   k1 => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=7] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 3, 3], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %select_scatter_default : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int, %select_1, 0, 0), kwargs = {})
#   %select_scatter_default_1 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %select_scatter_default, 0, 0), kwargs = {})
#   %select_scatter_default_2 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_1, %select_11, 0, 1), kwargs = {})
#   %select_scatter_default_3 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %select_scatter_default_2, 0, 1), kwargs = {})
#   %select_scatter_default_4 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_2, %select_23, 0, 2), kwargs = {})
#   %select_scatter_default_5 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_3, %select_scatter_default_4, 0, 2), kwargs = {})
triton_poi_fused_zeros_3 = async_compile.triton('triton_poi_fused_zeros_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 36
    x3 = (xindex % 36)
    x1 = ((xindex // 9) % 4)
    x0 = (xindex % 9)
    x5 = xindex
    tmp3 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr2 + (9 + x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp17 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = x1
    tmp7 = tmp6 == tmp4
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 0, tl.int32)
    tmp13 = tmp4 == tmp12
    tmp14 = tmp6 == tmp12
    tmp18 = tmp16 * tmp17
    tmp19 = 0.0
    tmp20 = tl.where(tmp14, tmp18, tmp19)
    tmp21 = tl.where(tmp13, tmp20, tmp19)
    tmp22 = tl.where(tmp7, tmp11, tmp21)
    tmp23 = tmp0 == tmp12
    tmp24 = tl.where(tmp23, tmp20, tmp19)
    tmp25 = tl.where(tmp5, tmp22, tmp24)
    tmp26 = tl.where(tmp2, tmp3, tmp25)
    tl.store(out_ptr0 + (x5), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ax/caxffsuog6ctbj3lnfpgn2iosv7ubx6kilwn7gq7i3dsdp2rohly.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %select_scatter_default_6 : [num_users=1] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_int_3, %select_35, 0, 3), kwargs = {})
#   %select_scatter_default_7 : [num_users=3] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_5, %select_scatter_default_6, 0, 3), kwargs = {})
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 36
    x1 = ((xindex // 9) % 4)
    x0 = (xindex % 9)
    x4 = (xindex % 36)
    x5 = xindex
    tmp5 = tl.load(in_ptr0 + (3))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (27 + x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (108 + x4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x5), xmask)
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x1
    tmp4 = tmp3 == tmp1
    tmp8 = tmp6 * tmp7
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tl.store(out_ptr0 + (x5), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j3/cj3jmajfuyo55uw2l5rmizt5sqfnzprh37acuvbl5ounbuzyhihi.py
# Topologically Sorted Source Nodes: [add_4, add_5, add_6, rep_weight_4], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   rep_weight_4 => add_7
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %convolution), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_2), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %convolution_4), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %convolution_6), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_poi_fused_add_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp7 = tl.load(in_ptr3 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yn/cynx3hh55ighh7tw3i2uvztwjw7hp4uqeh3lgcb23qrvx3itr6lr.py
# Topologically Sorted Source Nodes: [rep_bias_1, rep_bias_3, rep_bias_5, rep_bias_7, add_8, add_9, add_10, rep_bias_8], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_10 => add_10
#   add_8 => add_8
#   add_9 => add_9
#   rep_bias_1 => add
#   rep_bias_3 => add_1
#   rep_bias_5 => add_2
#   rep_bias_7 => add_3
#   rep_bias_8 => add_11
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_6), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %primals_10), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %primals_15), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_7, %primals_20), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %add), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %add_1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %add_2), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %add_3), kwargs = {})
triton_poi_fused_add_6 = async_compile.triton('triton_poi_fused_add_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp5 = tl.load(in_ptr2 + (x0), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask)
    tmp9 = tl.load(in_ptr4 + (x0), xmask)
    tmp10 = tl.load(in_ptr5 + (x0), xmask)
    tmp13 = tl.load(in_ptr6 + (x0), xmask)
    tmp14 = tl.load(in_ptr7 + (x0), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 + tmp11
    tmp15 = tmp13 + tmp14
    tmp16 = tmp12 + tmp15
    tl.store(in_out_ptr0 + (x0), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sc/cscs65ndtkflyknk2s2mbd45eulxlenelwdu3kizfv3ajisk52bn.py
# Topologically Sorted Source Nodes: [y, y_1], Original ATen: [aten.convolution, aten._prelu_kernel]
# Source node to ATen node mapping:
#   y => convolution_8
#   y_1 => gt, mul_7, where
# Graph fragment:
#   %convolution_8 : [num_users=4] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_22, %add_7, %add_11, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution_8, 0), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %convolution_8), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution_8, %mul_7), kwargs = {})
triton_poi_fused__prelu_kernel_convolution_7 = async_compile.triton('triton_poi_fused__prelu_kernel_convolution_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 16) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp6 = tmp5 * tmp2
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_4, (4, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_8, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_9, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_13, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_14, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_18, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_19, (4, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_23, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rep_weight], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(primals_3, buf0, 4, 4, grid=grid(4, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [rep_weight], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_4, buf0, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 3, 3), (36, 9, 3, 1))
        buf2 = empty_strided_cuda((1, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ones, rep_bias], Original ATen: [aten.ones, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_ones_1.run(primals_5, buf2, 36, grid=grid(36), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (1, 4, 1, 1), (4, 1, 1, 1))
        buf4 = empty_strided_cuda((4, 3, 3), (9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_8, primals_9, buf4, 36, grid=grid(36), stream=stream0)
        buf5 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [k1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_3.run(buf4, primals_8, primals_9, buf5, 144, grid=grid(144), stream=stream0)
        buf6 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_8, primals_9, buf5, buf6, 144, grid=grid(144), stream=stream0)
        del primals_8
        buf7 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [rep_weight_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(primals_7, buf7, 4, 4, grid=grid(4, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [rep_weight_1], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 4, 3, 3), (36, 9, 3, 1))
        buf9 = reinterpret_tensor(buf4, (1, 4, 3, 3), (36, 9, 3, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [ones, rep_bias_2], Original ATen: [aten.ones, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_ones_1.run(primals_11, buf9, 36, grid=grid(36), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, buf6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (1, 4, 1, 1), (4, 1, 1, 1))
        buf11 = empty_strided_cuda((4, 3, 3), (9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_13, primals_14, buf11, 36, grid=grid(36), stream=stream0)
        buf12 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [k1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_3.run(buf11, primals_13, primals_14, buf12, 144, grid=grid(144), stream=stream0)
        buf13 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_13, primals_14, buf12, buf13, 144, grid=grid(144), stream=stream0)
        del primals_13
        buf14 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [rep_weight_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(primals_12, buf14, 4, 4, grid=grid(4, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [rep_weight_2], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf13, buf14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 3, 3), (36, 9, 3, 1))
        buf16 = reinterpret_tensor(buf11, (1, 4, 3, 3), (36, 9, 3, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [ones, rep_bias_4], Original ATen: [aten.ones, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_ones_1.run(primals_16, buf16, 36, grid=grid(36), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [conv2d_5], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, buf13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (1, 4, 1, 1), (4, 1, 1, 1))
        buf18 = empty_strided_cuda((4, 3, 3), (9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_18, primals_19, buf18, 36, grid=grid(36), stream=stream0)
        buf19 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [k1], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_3.run(buf18, primals_18, primals_19, buf19, 144, grid=grid(144), stream=stream0)
        buf20 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_18, primals_19, buf19, buf20, 144, grid=grid(144), stream=stream0)
        del buf19
        del primals_18
        buf21 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [rep_weight_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(primals_17, buf21, 4, 4, grid=grid(4, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [rep_weight_3], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 4, 3, 3), (36, 9, 3, 1))
        del buf21
        buf23 = reinterpret_tensor(buf18, (1, 4, 3, 3), (36, 9, 3, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [ones, rep_bias_6], Original ATen: [aten.ones, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_ones_1.run(primals_21, buf23, 36, grid=grid(36), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [conv2d_7], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, buf20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (1, 4, 1, 1), (4, 1, 1, 1))
        buf25 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [add_4, add_5, add_6, rep_weight_4], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf25, primals_1, buf8, buf15, buf22, 144, grid=grid(144), stream=stream0)
        del buf15
        del buf22
        del buf8
        del primals_1
        buf26 = reinterpret_tensor(buf3, (4, ), (1, ), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [rep_bias_1, rep_bias_3, rep_bias_5, rep_bias_7, add_8, add_9, add_10, rep_bias_8], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6.run(buf26, primals_2, primals_6, buf10, primals_10, buf17, primals_15, buf24, primals_20, 4, grid=grid(4), stream=stream0)
        del buf10
        del buf17
        del buf24
        del primals_10
        del primals_15
        del primals_2
        del primals_20
        del primals_6
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(primals_22, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 4, 4, 4), (64, 16, 4, 1))
        buf28 = buf27; del buf27  # reuse
        buf29 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y, y_1], Original ATen: [aten.convolution, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_convolution_7.run(buf28, buf26, primals_23, buf29, 256, grid=grid(256), stream=stream0)
        del buf26
    return (buf29, primals_4, primals_9, primals_14, primals_19, primals_22, primals_23, reinterpret_tensor(primals_3, (4, 4, 1, 1), (1, 4, 1, 1), 0), buf2, buf6, reinterpret_tensor(primals_7, (4, 4, 1, 1), (1, 4, 1, 1), 0), buf9, buf13, reinterpret_tensor(primals_12, (4, 4, 1, 1), (1, 4, 1, 1), 0), buf16, buf20, reinterpret_tensor(primals_17, (4, 4, 1, 1), (1, 4, 1, 1), 0), buf23, buf25, buf28, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
