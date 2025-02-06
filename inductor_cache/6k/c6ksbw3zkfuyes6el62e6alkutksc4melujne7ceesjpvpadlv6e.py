# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/jw/cjwyrurcbwfi3fxbiwjozeulsvtjoe7iytcikygped3mpl37tpxs.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6l/c6lwewpurffoyufws522n2fexr44nrocozk7vqfifkrtuzw3pg6d.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2112
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 33)
    y1 = yindex // 33
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 33*x2 + 297*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/rf/crf3ta5u4jvgjjek4n6pmqx2ufr472nmfltri7zjb3uefhrusz6q.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16512
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 129)
    y1 = yindex // 129
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 129*x2 + 1161*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/c2/cc23w3dqxlfxxzs2cccicldgnszcm54ksd3hmgkus6yibhroafv3.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   output => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_2, %primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 32768*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/cw/ccweb7ojbx3moudyv2ty7haiz3r3huf5svngcji53utn7x52kx5p.py
# Topologically Sorted Source Nodes: [output_1, output_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_1 => add_1, mul_1, mul_2, sub
#   output_2 => gt, mul_3, where
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view, %add_1), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/md/cmda7leyca2kmvjbk74n46fsn26pvidvaljse474dvcncqvczv2q.py
# Topologically Sorted Source Nodes: [output_7, output_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_7 => add_5, mul_10, mul_9, sub_2
#   output_8 => gt_2, mul_11, where_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_5, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %add_5), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %add_5, %mul_11), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 32}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 32
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y2 = (yindex % 1024)
    y3 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x1 + 32*y0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = tmp15 > tmp16
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(out_ptr1 + (y2 + 1024*x1 + 33792*y3), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h6/ch6krgkba3kfkg6ajuuozke3xxua7a7pvokjhi2fwuyxpkoapfnb.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%primals_2, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_6 = async_compile.triton('triton_poi_fused_avg_pool2d_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 32) % 32)
    x0 = (xindex % 32)
    x3 = xindex // 32
    x2 = xindex // 1024
    x4 = (xindex % 1024)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-65) + 2*x0 + 128*x3), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-64) + 2*x0 + 128*x3), tmp16, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-63) + 2*x0 + 128*x3), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 128*x3), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 128*x3), tmp33, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 128*x3), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (63 + 2*x0 + 128*x3), tmp43, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (64 + 2*x0 + 128*x3), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (65 + 2*x0 + 128*x3), tmp49, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((65) * ((65) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (65)))*((65) * ((65) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (65))) + ((-2)*x0*((65) * ((65) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (65)))) + ((-2)*x1*((65) * ((65) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (65)))) + 4*x0*x1 + ((65) * ((65) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (65))) + ((65) * ((65) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (65)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4 + 33792*x2), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/b6/cb6gbehist3bao4cctxsaqfn4ys433k6xedyxandiujhp64krxcj.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_3 => avg_pool2d_2
# Graph fragment:
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_7 = async_compile.triton('triton_poi_fused_avg_pool2d_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + 2*x0 + 64*x1 + 33792*x2), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-32) + 2*x0 + 64*x1 + 33792*x2), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-31) + 2*x0 + 64*x1 + 33792*x2), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 64*x1 + 33792*x2), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 64*x1 + 33792*x2), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 64*x1 + 33792*x2), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (31 + 2*x0 + 64*x1 + 33792*x2), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (32 + 2*x0 + 64*x1 + 33792*x2), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (33 + 2*x0 + 64*x1 + 33792*x2), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((33) * ((33) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (33)))*((33) * ((33) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (33))) + ((-2)*x0*((33) * ((33) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (33)))) + ((-2)*x1*((33) * ((33) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (33)))) + 4*x0*x1 + ((33) * ((33) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (33))) + ((33) * ((33) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (33)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (129*x4), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4m/c4m64rztci4alfctmp252zk7n5sqktnjjoh7tnftq7j4a4uxhrg6.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%where_2, %avg_pool2d], 1), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 1024}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 132
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 33)
    y1 = yindex // 33
    tmp0 = tl.load(in_ptr0 + (x2 + 1024*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 33*x2 + 33792*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/33/c33apo2uybv6hbv7px54jwlegl7bard4bvcrkthrhoc2jtnkavcn.py
# Topologically Sorted Source Nodes: [output_9, output_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_10 => gt_3, mul_15, where_3
#   output_9 => add_7, mul_13, mul_14, sub_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat, %unsqueeze_25), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_31), kwargs = {})
#   %gt_3 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_7, 0), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_3, %add_7), kwargs = {})
#   %where_3 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_3, %add_7, %mul_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 135168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 33)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/hg/chg52aif5ejxqratbfpezl5bg4t4b5j4qi47tmprnjrub7ryacmf.py
# Topologically Sorted Source Nodes: [output_12, output_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_12 => add_9, mul_17, mul_18, sub_4
#   output_13 => gt_4, mul_19, where_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_33), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %unsqueeze_39), kwargs = {})
#   %gt_4 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_9, 0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %add_9), kwargs = {})
#   %where_4 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_9, %mul_19), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/pa/cpa65stvc4kvon5qdrrrksm2omx2xffcohnkgh6j35crmb7tj7u6.py
# Topologically Sorted Source Nodes: [joi_feat, joi_feat_1, joi_feat_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   joi_feat => cat_1
#   joi_feat_1 => add_11, mul_21, mul_22, sub_5
#   joi_feat_2 => gt_5, mul_23, where_5
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_4, %convolution_5], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_1, %unsqueeze_41), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, %unsqueeze_47), kwargs = {})
#   %gt_5 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_11, 0), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %add_11), kwargs = {})
#   %where_5 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_5, %add_11, %mul_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (64*x1 + ((-64) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp29 = tmp28 * tmp25
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(in_out_ptr0 + (x2), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/sz/csz6kbmdw4kxxsnxk7fxud3xfqli5hzff6v7nras7qyddxgbkq7i.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_6, [-1, -2], True), kwargs = {})
triton_red_fused_mean_12 = async_compile.triton('triton_red_fused_mean_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_12(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 64)
    x1 = xindex // 64
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 8192*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pm/cpmto4yvipd2op33ww5z4fur45w3xc4w6cuo4orcs4kvh2ix23by.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_6, [-1, -2], True), kwargs = {})
triton_per_fused_mean_13 = async_compile.triton('triton_per_fused_mean_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 2},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_13(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2
    RBLOCK: tl.constexpr = 2
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
    tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 128*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 256.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wx/cwxmo57u2cddbign3d345ufqngm564ec7zcl7mf45dsz3n2225ca.py
# Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_4 => add_tensor_23
#   input_5 => relu
# Graph fragment:
#   %add_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %primals_40), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_23,), kwargs = {})
triton_poi_fused_addmm_relu_14 = async_compile.triton('triton_poi_fused_addmm_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqshfy4uni5z3hji666ftui3i4op3cyeiqmirfurclwweou4vpgj.py
# Topologically Sorted Source Nodes: [output_17], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   output_17 => mul_24
# Graph fragment:
#   %mul_24 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_6, %view_7), kwargs = {})
triton_poi_fused_mul_15 = async_compile.triton('triton_poi_fused_mul_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/l2/cl2zqr47te25rep6xdx3r2chumeod5em3qkoenzo3ds3n4wa4g3u.py
# Topologically Sorted Source Nodes: [output_19, output_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_19 => add_13, mul_26, mul_27, sub_6
#   output_20 => gt_6, mul_28, where_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_49), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_55), kwargs = {})
#   %gt_6 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_13, 0), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, %add_13), kwargs = {})
#   %where_6 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_6, %add_13, %mul_28), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/4i/c4iy3uz4yd4rhso6a3x2d7tcjdby7an3w7jr2hszulccsnzpzhuj.py
# Topologically Sorted Source Nodes: [joi_feat_3, output_23], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   joi_feat_3 => cat_2
#   output_23 => add_15, mul_30, mul_31, sub_7
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_8, %convolution_9], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_2, %unsqueeze_57), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_63), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (32*x1 + ((-32) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/rm/crmnp3u4yec5xne7ilibmsgpfxdhqaqbccpthefgsst3z4h6x45u.py
# Topologically Sorted Source Nodes: [output_24, adaptive_avg_pool2d_1], Original ATen: [aten._prelu_kernel, aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d_1 => mean_1
#   output_24 => gt_7, mul_32, where_7
# Graph fragment:
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_15, 0), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %add_15), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %add_15, %mul_32), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%where_7, [-1, -2], True), kwargs = {})
triton_red_fused__prelu_kernel_mean_18 = async_compile.triton('triton_red_fused__prelu_kernel_mean_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__prelu_kernel_mean_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__prelu_kernel_mean_18(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 64*r2 + 8192*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = 0.0
        tmp2 = tmp0 > tmp1
        tmp4 = tmp3 * tmp0
        tmp5 = tl.where(tmp2, tmp0, tmp4)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nh/cnhpbslbcp3owelra62gaspao6tugipqn6zzpjmqxeuxqnkt5o6c.py
# Topologically Sorted Source Nodes: [output_24, output_25, output_26, cat_4], Original ATen: [aten._prelu_kernel, aten.mul, aten.add, aten.cat]
# Source node to ATen node mapping:
#   cat_4 => cat_4
#   output_24 => gt_7, mul_32, where_7
#   output_25 => mul_33
#   output_26 => add_16
# Graph fragment:
#   %gt_7 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_15, 0), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %add_15), kwargs = {})
#   %where_7 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_7, %add_15, %mul_32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_7, %view_11), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %mul_33), kwargs = {})
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%add_21, %mul_24, %avg_pool2d_2], 1), kwargs = {})
triton_poi_fused__prelu_kernel_add_cat_mul_19 = async_compile.triton('triton_poi_fused__prelu_kernel_add_cat_mul_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_cat_mul_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_cat_mul_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 16384
    x3 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_out_ptr0 + (x4), None)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(in_out_ptr0 + (x4), tmp10, None)
    tl.store(out_ptr0 + (x0 + 129*x3), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/bt/cbtwhyojip34m75mp7lo77rczaq2ek4hnu2ieysohhy2iviwamul.py
# Topologically Sorted Source Nodes: [output_33, output_34, output_35], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
# Source node to ATen node mapping:
#   output_33 => gt_9, mul_41, where_9
#   output_34 => mul_42
#   output_35 => add_21
# Graph fragment:
#   %gt_9 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_20, 0), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %add_20), kwargs = {})
#   %where_9 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_9, %add_20, %mul_41), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_9, %view_15), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %mul_42), kwargs = {})
triton_poi_fused__prelu_kernel_add_mul_20 = async_compile.triton('triton_poi_fused__prelu_kernel_add_mul_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_mul_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_mul_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 16384
    x4 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0 + 64*x2), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(out_ptr0 + (x0 + 129*x4), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/tf/ctfaatzo54u3jvcqtazhveciwpynzfxof4xao33pvwp56kthls4g.py
# Topologically Sorted Source Nodes: [output_36, output_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_36 => add_23, mul_44, mul_45, sub_10
#   output_37 => gt_10, mul_46, where_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_4, %unsqueeze_81), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %unsqueeze_85), kwargs = {})
#   %add_23 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %unsqueeze_87), kwargs = {})
#   %gt_10 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_23, 0), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, %add_23), kwargs = {})
#   %where_10 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_10, %add_23, %mul_46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 132096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 129)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n5/cn5ombv7nazludgxm57h5ozic5iqqhany42ycs5telyxq2ffujyz.py
# Topologically Sorted Source Nodes: [output_39, output_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_39 => add_25, mul_48, mul_49, sub_11
#   output_40 => gt_11, mul_50, where_11
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_89), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %unsqueeze_93), kwargs = {})
#   %add_25 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %unsqueeze_95), kwargs = {})
#   %gt_11 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_25, 0), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %add_25), kwargs = {})
#   %where_11 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_11, %add_25, %mul_50), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/de/cdeejvnbiiq5z7yz36glbz2f723ohxtuh5ovykm4hqcrlf3mtlgn.py
# Topologically Sorted Source Nodes: [joi_feat_5, joi_feat_6, joi_feat_7], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   joi_feat_5 => cat_5
#   joi_feat_6 => add_27, mul_52, mul_53, sub_12
#   joi_feat_7 => gt_12, mul_54, where_12
# Graph fragment:
#   %cat_5 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_14, %convolution_15], 1), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_5, %unsqueeze_97), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_101), kwargs = {})
#   %add_27 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_103), kwargs = {})
#   %gt_12 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_27, 0), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_18, %add_27), kwargs = {})
#   %where_12 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_27, %mul_54), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128*x1 + ((-128) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = 0.0
    tmp27 = tmp25 > tmp26
    tmp29 = tmp28 * tmp25
    tmp30 = tl.where(tmp27, tmp25, tmp29)
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(in_out_ptr0 + (x2), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/o3/co3gksmnzhzsbzrpjmvx5jsw2fcctg3fagy5gio2c4ypoivl3kn3.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d_3], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d_3 => mean_3
# Graph fragment:
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%convolution_16, [-1, -2], True), kwargs = {})
triton_per_fused_mean_24 = async_compile.triton('triton_per_fused_mean_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_24(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 8192*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/no/cnohsky3ivrs44mxyceixq5fsqfh4kv3uvnlsqien6ysxslcwtgf.py
# Topologically Sorted Source Nodes: [output_44], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   output_44 => mul_55
# Graph fragment:
#   %mul_55 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution_16, %view_20), kwargs = {})
triton_poi_fused_mul_25 = async_compile.triton('triton_poi_fused_mul_25', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_25(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(out_ptr0 + (x3), tmp3, None)
''', device_str='cuda')


# kernel path: inductor_cache/bp/cbp2itpvn6ytowv3ztym3fmr4mgmwgilq6ghbbaccx6dxhvyrojk.py
# Topologically Sorted Source Nodes: [output_46, output_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   output_46 => add_29, mul_57, mul_58, sub_13
#   output_47 => gt_13, mul_59, where_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_105), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %unsqueeze_109), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_58, %unsqueeze_111), kwargs = {})
#   %gt_13 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_29, 0), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, %add_29), kwargs = {})
#   %where_13 : [num_users=3] = call_function[target=torch.ops.aten.where.self](args = (%gt_13, %add_29, %mul_59), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
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
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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
    tmp19 = tmp18 * tmp15
    tmp20 = tl.where(tmp17, tmp15, tmp19)
    tl.store(in_out_ptr0 + (x2), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/6k/c6kokvmbkprkm74nablvr36yk4wwjplog2tkwpfnsde6dl3c7w2r.py
# Topologically Sorted Source Nodes: [joi_feat_8, output_50], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   joi_feat_8 => cat_6
#   output_50 => add_31, mul_61, mul_62, sub_14
# Graph fragment:
#   %cat_6 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_18, %convolution_19], 1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_6, %unsqueeze_113), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_117), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_119), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_cat_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_cat_27', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 128, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (64*x1 + ((-64) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = tmp10 - tmp11
    tmp14 = 0.001
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.sqrt(tmp15)
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp17 / tmp16
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp12 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x2), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/as/casxjhug4s67wmm4hbvcojrjy6c6tp3to2heigp6clbsbhku26nx.py
# Topologically Sorted Source Nodes: [output_51, adaptive_avg_pool2d_4], Original ATen: [aten._prelu_kernel, aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d_4 => mean_4
#   output_51 => gt_14, mul_63, where_14
# Graph fragment:
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 0), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %add_31), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %add_31, %mul_63), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%where_14, [-1, -2], True), kwargs = {})
triton_per_fused__prelu_kernel_mean_28 = async_compile.triton('triton_per_fused__prelu_kernel_mean_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 512, 'r': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__prelu_kernel_mean_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__prelu_kernel_mean_28(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*r2 + 8192*x1), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp4 = tmp3 * tmp0
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = 64.0
    tmp11 = tmp9 / tmp10
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7j/c7j24vmbpb72im7x64scw3p6pcrtdz6t5qeo6bddkvs6tniwbvrq.py
# Topologically Sorted Source Nodes: [output_51, output_52, output_53], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
# Source node to ATen node mapping:
#   output_51 => gt_14, mul_63, where_14
#   output_52 => mul_64
#   output_53 => add_32
# Graph fragment:
#   %gt_14 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_31, 0), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_22, %add_31), kwargs = {})
#   %where_14 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_14, %add_31, %mul_63), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_14, %view_24), kwargs = {})
#   %add_32 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %mul_64), kwargs = {})
triton_poi_fused__prelu_kernel_add_mul_29 = async_compile.triton('triton_poi_fused__prelu_kernel_add_mul_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__prelu_kernel_add_mul_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__prelu_kernel_add_mul_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 128)
    x2 = xindex // 8192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_out_ptr0 + (x3), None)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x0 + 128*x2), None, eviction_policy='evict_last')
    tmp2 = 0.0
    tmp3 = tmp1 > tmp2
    tmp5 = tmp4 * tmp1
    tmp6 = tl.where(tmp3, tmp1, tmp5)
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp0 + tmp9
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/lc/clcl6zdbvhmqoagcrev7ttvkldhg4mdm3s7zuq2kiieeg43edfhx.py
# Topologically Sorted Source Nodes: [cat_26, output_225, output_226], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
# Source node to ATen node mapping:
#   cat_26 => cat_26
#   output_225 => add_129, mul_237, mul_238, sub_53
#   output_226 => gt_53, mul_239, where_53
# Graph fragment:
#   %cat_26 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_55, %add_127], 1), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_26, %unsqueeze_425), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_238 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_237, %unsqueeze_429), kwargs = {})
#   %add_129 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_238, %unsqueeze_431), kwargs = {})
#   %gt_53 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_129, 0), kwargs = {})
#   %mul_239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_101, %add_129), kwargs = {})
#   %where_53 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_53, %add_129, %mul_239), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x3 = xindex // 256
    x2 = xindex // 16384
    x4 = xindex
    tmp23 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 256, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (128*x3 + ((-128) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (128*x3 + ((-128) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp11 = 0.0
    tmp12 = tmp10 > tmp11
    tmp13 = tl.load(in_ptr3 + ((-128) + x0), tmp6, eviction_policy='evict_last', other=0.0)
    tmp14 = tmp13 * tmp10
    tmp15 = tl.where(tmp12, tmp10, tmp14)
    tmp16 = tl.load(in_ptr4 + (128*x2 + ((-128) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp15 * tmp17
    tmp19 = tmp9 + tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp6, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.sqrt(tmp27)
    tmp29 = tl.full([1], 1, tl.int32)
    tmp30 = tmp29 / tmp28
    tmp31 = 1.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = 0.0
    tmp39 = tmp37 > tmp38
    tmp41 = tmp40 * tmp37
    tmp42 = tl.where(tmp39, tmp37, tmp41)
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(in_out_ptr0 + (x4), tmp42, None)
''', device_str='cuda')


# kernel path: inductor_cache/ac/cacjv63if6besndvbmt2aetgiizlzheky2zpn6jkmza4ag4cc3fe.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   out => convert_element_type_109
# Graph fragment:
#   %convert_element_type_109 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_102, torch.int64), kwargs = {})
triton_poi_fused__to_copy_31 = async_compile.triton('triton_poi_fused__to_copy_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_31(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fy/cfy2jwafroddgjcsrwdh47jwmg74gy372gtspsyljp7ut3qnwriq.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   out => add_131, clamp_max
# Graph fragment:
#   %add_131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_109, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_131, 7), kwargs = {})
triton_poi_fused_add_clamp_32 = async_compile.triton('triton_poi_fused_add_clamp_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_32(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 7, tl.int64)
    tmp13 = triton_helpers.minimum(tmp11, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5a/c5ahkcua3s2mowayya6mdmrctkq2vh3rupel2dj37ozbtgdi3mvr.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   out => add_130, clamp_max_2, clamp_min, clamp_min_2, convert_element_type_108, iota, mul_240, sub_54, sub_56
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_108 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add_130 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_108, 0.5), kwargs = {})
#   %mul_240 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_130, 0.125), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_240, 0.5), kwargs = {})
#   %clamp_min : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_54, 0.0), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min, %convert_element_type_111), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_56, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_mul_sub_33 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_mul_sub_33', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_mul_sub_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_mul_sub_33(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = 0.0
    tmp8 = triton_helpers.maximum(tmp6, tmp7)
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 - tmp10
    tmp12 = triton_helpers.maximum(tmp11, tmp7)
    tmp13 = 1.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/za/cza5t2drcvhkblj3bpt4mizlnjr5jddeexe44vu54o7csmvxxexc.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   out => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_134, add_135, add_136, mul_242, mul_243, mul_244, sub_57, sub_58, sub_60
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_77, [None, None, %convert_element_type_109, %convert_element_type_111]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_77, [None, None, %convert_element_type_109, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_77, [None, None, %clamp_max, %convert_element_type_111]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_77, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_242 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %clamp_max_2), kwargs = {})
#   %add_134 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_242), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %clamp_max_2), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_243), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_135, %add_134), kwargs = {})
#   %mul_244 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_60, %clamp_max_3), kwargs = {})
#   %add_136 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_134, %mul_244), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_34 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 311296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x2 = ((xindex // 4096) % 19)
    x3 = xindex // 77824
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 8, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (x2 + 19*tmp8 + 152*tmp4 + 1216*x3), None, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (x2 + 19*tmp13 + 152*tmp4 + 1216*x3), None, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp20 = tmp19 + tmp1
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tmp23 = tl.load(in_ptr2 + (x2 + 19*tmp8 + 152*tmp22 + 1216*x3), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x2 + 19*tmp13 + 152*tmp22 + 1216*x3), None, eviction_policy='evict_last')
    tmp25 = tmp24 - tmp23
    tmp26 = tmp25 * tmp16
    tmp27 = tmp23 + tmp26
    tmp28 = tmp27 - tmp18
    tmp30 = tmp28 * tmp29
    tmp31 = tmp18 + tmp30
    tl.store(in_out_ptr0 + (x5), tmp31, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445 = args
    args.clear()
    assert_size_stride(primals_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_2, (4, 1, 64, 64), (4096, 4096, 64, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, ), (1, ))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (33, ), (1, ))
    assert_size_stride(primals_21, (33, ), (1, ))
    assert_size_stride(primals_22, (33, ), (1, ))
    assert_size_stride(primals_23, (33, ), (1, ))
    assert_size_stride(primals_24, (33, ), (1, ))
    assert_size_stride(primals_25, (64, 33, 3, 3), (297, 9, 3, 1))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, ), (1, ))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_32, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_39, (8, 64), (64, 1))
    assert_size_stride(primals_40, (8, ), (1, ))
    assert_size_stride(primals_41, (64, 8), (8, 1))
    assert_size_stride(primals_42, (64, ), (1, ))
    assert_size_stride(primals_43, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, ), (1, ))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_50, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (64, ), (1, ))
    assert_size_stride(primals_52, (64, ), (1, ))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (8, 64), (64, 1))
    assert_size_stride(primals_57, (8, ), (1, ))
    assert_size_stride(primals_58, (64, 8), (8, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (32, ), (1, ))
    assert_size_stride(primals_63, (32, ), (1, ))
    assert_size_stride(primals_64, (32, ), (1, ))
    assert_size_stride(primals_65, (32, ), (1, ))
    assert_size_stride(primals_66, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_67, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (64, ), (1, ))
    assert_size_stride(primals_69, (64, ), (1, ))
    assert_size_stride(primals_70, (64, ), (1, ))
    assert_size_stride(primals_71, (64, ), (1, ))
    assert_size_stride(primals_72, (64, ), (1, ))
    assert_size_stride(primals_73, (8, 64), (64, 1))
    assert_size_stride(primals_74, (8, ), (1, ))
    assert_size_stride(primals_75, (64, 8), (8, 1))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (129, ), (1, ))
    assert_size_stride(primals_78, (129, ), (1, ))
    assert_size_stride(primals_79, (129, ), (1, ))
    assert_size_stride(primals_80, (129, ), (1, ))
    assert_size_stride(primals_81, (129, ), (1, ))
    assert_size_stride(primals_82, (128, 129, 3, 3), (1161, 9, 3, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, ), (1, ))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_89, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (256, ), (1, ))
    assert_size_stride(primals_93, (256, ), (1, ))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_96, (8, 128), (128, 1))
    assert_size_stride(primals_97, (8, ), (1, ))
    assert_size_stride(primals_98, (128, 8), (8, 1))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (64, ), (1, ))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (64, ), (1, ))
    assert_size_stride(primals_105, (64, ), (1, ))
    assert_size_stride(primals_106, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (128, ), (1, ))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (8, 128), (128, 1))
    assert_size_stride(primals_114, (8, ), (1, ))
    assert_size_stride(primals_115, (128, 8), (8, 1))
    assert_size_stride(primals_116, (128, ), (1, ))
    assert_size_stride(primals_117, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_118, (64, ), (1, ))
    assert_size_stride(primals_119, (64, ), (1, ))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, ), (1, ))
    assert_size_stride(primals_123, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_124, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (8, 128), (128, 1))
    assert_size_stride(primals_131, (8, ), (1, ))
    assert_size_stride(primals_132, (128, 8), (8, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_135, (64, ), (1, ))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, ), (1, ))
    assert_size_stride(primals_138, (64, ), (1, ))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_141, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (8, 128), (128, 1))
    assert_size_stride(primals_148, (8, ), (1, ))
    assert_size_stride(primals_149, (128, 8), (8, 1))
    assert_size_stride(primals_150, (128, ), (1, ))
    assert_size_stride(primals_151, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_152, (64, ), (1, ))
    assert_size_stride(primals_153, (64, ), (1, ))
    assert_size_stride(primals_154, (64, ), (1, ))
    assert_size_stride(primals_155, (64, ), (1, ))
    assert_size_stride(primals_156, (64, ), (1, ))
    assert_size_stride(primals_157, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (8, 128), (128, 1))
    assert_size_stride(primals_165, (8, ), (1, ))
    assert_size_stride(primals_166, (128, 8), (8, 1))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_168, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (64, ), (1, ))
    assert_size_stride(primals_170, (64, ), (1, ))
    assert_size_stride(primals_171, (64, ), (1, ))
    assert_size_stride(primals_172, (64, ), (1, ))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_175, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (8, 128), (128, 1))
    assert_size_stride(primals_182, (8, ), (1, ))
    assert_size_stride(primals_183, (128, 8), (8, 1))
    assert_size_stride(primals_184, (128, ), (1, ))
    assert_size_stride(primals_185, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_186, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_189, (64, ), (1, ))
    assert_size_stride(primals_190, (64, ), (1, ))
    assert_size_stride(primals_191, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_192, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (128, ), (1, ))
    assert_size_stride(primals_195, (128, ), (1, ))
    assert_size_stride(primals_196, (128, ), (1, ))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_198, (8, 128), (128, 1))
    assert_size_stride(primals_199, (8, ), (1, ))
    assert_size_stride(primals_200, (128, 8), (8, 1))
    assert_size_stride(primals_201, (128, ), (1, ))
    assert_size_stride(primals_202, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_203, (64, ), (1, ))
    assert_size_stride(primals_204, (64, ), (1, ))
    assert_size_stride(primals_205, (64, ), (1, ))
    assert_size_stride(primals_206, (64, ), (1, ))
    assert_size_stride(primals_207, (64, ), (1, ))
    assert_size_stride(primals_208, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_209, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_210, (128, ), (1, ))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_214, (128, ), (1, ))
    assert_size_stride(primals_215, (8, 128), (128, 1))
    assert_size_stride(primals_216, (8, ), (1, ))
    assert_size_stride(primals_217, (128, 8), (8, 1))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (64, ), (1, ))
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_226, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_227, (128, ), (1, ))
    assert_size_stride(primals_228, (128, ), (1, ))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (128, ), (1, ))
    assert_size_stride(primals_232, (8, 128), (128, 1))
    assert_size_stride(primals_233, (8, ), (1, ))
    assert_size_stride(primals_234, (128, 8), (8, 1))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_237, (64, ), (1, ))
    assert_size_stride(primals_238, (64, ), (1, ))
    assert_size_stride(primals_239, (64, ), (1, ))
    assert_size_stride(primals_240, (64, ), (1, ))
    assert_size_stride(primals_241, (64, ), (1, ))
    assert_size_stride(primals_242, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_243, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_246, (128, ), (1, ))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (8, 128), (128, 1))
    assert_size_stride(primals_250, (8, ), (1, ))
    assert_size_stride(primals_251, (128, 8), (8, 1))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (64, ), (1, ))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_260, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (128, ), (1, ))
    assert_size_stride(primals_265, (128, ), (1, ))
    assert_size_stride(primals_266, (8, 128), (128, 1))
    assert_size_stride(primals_267, (8, ), (1, ))
    assert_size_stride(primals_268, (128, 8), (8, 1))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_271, (64, ), (1, ))
    assert_size_stride(primals_272, (64, ), (1, ))
    assert_size_stride(primals_273, (64, ), (1, ))
    assert_size_stride(primals_274, (64, ), (1, ))
    assert_size_stride(primals_275, (64, ), (1, ))
    assert_size_stride(primals_276, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_277, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_278, (128, ), (1, ))
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_283, (8, 128), (128, 1))
    assert_size_stride(primals_284, (8, ), (1, ))
    assert_size_stride(primals_285, (128, 8), (8, 1))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_288, (64, ), (1, ))
    assert_size_stride(primals_289, (64, ), (1, ))
    assert_size_stride(primals_290, (64, ), (1, ))
    assert_size_stride(primals_291, (64, ), (1, ))
    assert_size_stride(primals_292, (64, ), (1, ))
    assert_size_stride(primals_293, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_294, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (8, 128), (128, 1))
    assert_size_stride(primals_301, (8, ), (1, ))
    assert_size_stride(primals_302, (128, 8), (8, 1))
    assert_size_stride(primals_303, (128, ), (1, ))
    assert_size_stride(primals_304, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_305, (64, ), (1, ))
    assert_size_stride(primals_306, (64, ), (1, ))
    assert_size_stride(primals_307, (64, ), (1, ))
    assert_size_stride(primals_308, (64, ), (1, ))
    assert_size_stride(primals_309, (64, ), (1, ))
    assert_size_stride(primals_310, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_311, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (128, ), (1, ))
    assert_size_stride(primals_315, (128, ), (1, ))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (8, 128), (128, 1))
    assert_size_stride(primals_318, (8, ), (1, ))
    assert_size_stride(primals_319, (128, 8), (8, 1))
    assert_size_stride(primals_320, (128, ), (1, ))
    assert_size_stride(primals_321, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_322, (64, ), (1, ))
    assert_size_stride(primals_323, (64, ), (1, ))
    assert_size_stride(primals_324, (64, ), (1, ))
    assert_size_stride(primals_325, (64, ), (1, ))
    assert_size_stride(primals_326, (64, ), (1, ))
    assert_size_stride(primals_327, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_328, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (128, ), (1, ))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (128, ), (1, ))
    assert_size_stride(primals_333, (128, ), (1, ))
    assert_size_stride(primals_334, (8, 128), (128, 1))
    assert_size_stride(primals_335, (8, ), (1, ))
    assert_size_stride(primals_336, (128, 8), (8, 1))
    assert_size_stride(primals_337, (128, ), (1, ))
    assert_size_stride(primals_338, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_339, (64, ), (1, ))
    assert_size_stride(primals_340, (64, ), (1, ))
    assert_size_stride(primals_341, (64, ), (1, ))
    assert_size_stride(primals_342, (64, ), (1, ))
    assert_size_stride(primals_343, (64, ), (1, ))
    assert_size_stride(primals_344, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_345, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (128, ), (1, ))
    assert_size_stride(primals_349, (128, ), (1, ))
    assert_size_stride(primals_350, (128, ), (1, ))
    assert_size_stride(primals_351, (8, 128), (128, 1))
    assert_size_stride(primals_352, (8, ), (1, ))
    assert_size_stride(primals_353, (128, 8), (8, 1))
    assert_size_stride(primals_354, (128, ), (1, ))
    assert_size_stride(primals_355, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_356, (64, ), (1, ))
    assert_size_stride(primals_357, (64, ), (1, ))
    assert_size_stride(primals_358, (64, ), (1, ))
    assert_size_stride(primals_359, (64, ), (1, ))
    assert_size_stride(primals_360, (64, ), (1, ))
    assert_size_stride(primals_361, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_362, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (128, ), (1, ))
    assert_size_stride(primals_366, (128, ), (1, ))
    assert_size_stride(primals_367, (128, ), (1, ))
    assert_size_stride(primals_368, (8, 128), (128, 1))
    assert_size_stride(primals_369, (8, ), (1, ))
    assert_size_stride(primals_370, (128, 8), (8, 1))
    assert_size_stride(primals_371, (128, ), (1, ))
    assert_size_stride(primals_372, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_373, (64, ), (1, ))
    assert_size_stride(primals_374, (64, ), (1, ))
    assert_size_stride(primals_375, (64, ), (1, ))
    assert_size_stride(primals_376, (64, ), (1, ))
    assert_size_stride(primals_377, (64, ), (1, ))
    assert_size_stride(primals_378, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_379, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_380, (128, ), (1, ))
    assert_size_stride(primals_381, (128, ), (1, ))
    assert_size_stride(primals_382, (128, ), (1, ))
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_384, (128, ), (1, ))
    assert_size_stride(primals_385, (8, 128), (128, 1))
    assert_size_stride(primals_386, (8, ), (1, ))
    assert_size_stride(primals_387, (128, 8), (8, 1))
    assert_size_stride(primals_388, (128, ), (1, ))
    assert_size_stride(primals_389, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_390, (64, ), (1, ))
    assert_size_stride(primals_391, (64, ), (1, ))
    assert_size_stride(primals_392, (64, ), (1, ))
    assert_size_stride(primals_393, (64, ), (1, ))
    assert_size_stride(primals_394, (64, ), (1, ))
    assert_size_stride(primals_395, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_396, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_397, (128, ), (1, ))
    assert_size_stride(primals_398, (128, ), (1, ))
    assert_size_stride(primals_399, (128, ), (1, ))
    assert_size_stride(primals_400, (128, ), (1, ))
    assert_size_stride(primals_401, (128, ), (1, ))
    assert_size_stride(primals_402, (8, 128), (128, 1))
    assert_size_stride(primals_403, (8, ), (1, ))
    assert_size_stride(primals_404, (128, 8), (8, 1))
    assert_size_stride(primals_405, (128, ), (1, ))
    assert_size_stride(primals_406, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_407, (64, ), (1, ))
    assert_size_stride(primals_408, (64, ), (1, ))
    assert_size_stride(primals_409, (64, ), (1, ))
    assert_size_stride(primals_410, (64, ), (1, ))
    assert_size_stride(primals_411, (64, ), (1, ))
    assert_size_stride(primals_412, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_413, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_414, (128, ), (1, ))
    assert_size_stride(primals_415, (128, ), (1, ))
    assert_size_stride(primals_416, (128, ), (1, ))
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_418, (128, ), (1, ))
    assert_size_stride(primals_419, (8, 128), (128, 1))
    assert_size_stride(primals_420, (8, ), (1, ))
    assert_size_stride(primals_421, (128, 8), (8, 1))
    assert_size_stride(primals_422, (128, ), (1, ))
    assert_size_stride(primals_423, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_424, (64, ), (1, ))
    assert_size_stride(primals_425, (64, ), (1, ))
    assert_size_stride(primals_426, (64, ), (1, ))
    assert_size_stride(primals_427, (64, ), (1, ))
    assert_size_stride(primals_428, (64, ), (1, ))
    assert_size_stride(primals_429, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_430, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, ), (1, ))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (128, ), (1, ))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (8, 128), (128, 1))
    assert_size_stride(primals_437, (8, ), (1, ))
    assert_size_stride(primals_438, (128, 8), (8, 1))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (256, ), (1, ))
    assert_size_stride(primals_441, (256, ), (1, ))
    assert_size_stride(primals_442, (256, ), (1, ))
    assert_size_stride(primals_443, (256, ), (1, ))
    assert_size_stride(primals_444, (256, ), (1, ))
    assert_size_stride(primals_445, (19, 256, 1, 1), (256, 1, 1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_8, buf0, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_8
        buf1 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_14, buf1, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_14
        buf2 = empty_strided_cuda((64, 33, 3, 3), (297, 1, 99, 33), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_25, buf2, 2112, 9, grid=grid(2112, 9), stream=stream0)
        del primals_25
        buf3 = empty_strided_cuda((128, 129, 3, 3), (1161, 1, 387, 129), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_82, buf3, 16512, 9, grid=grid(16512, 9), stream=stream0)
        del primals_82
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 32, 32, 32), (32768, 1024, 32, 1))
        buf5 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf4, buf5, 128, 1024, grid=grid(128, 1024), stream=stream0)
        buf6 = reinterpret_tensor(buf4, (4, 32, 32, 32), (32768, 1, 1024, 32), 0); del buf4  # reuse
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [output_1, output_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_4.run(buf7, buf5, primals_3, primals_4, primals_5, primals_6, primals_7, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [output_3], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf9 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [output_4, output_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_4.run(buf10, buf8, primals_9, primals_10, primals_11, primals_12, primals_13, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [output_6], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf16 = empty_strided_cuda((4, 33, 32, 32), (33792, 1024, 32, 1), torch.float32)
        buf15 = reinterpret_tensor(buf16, (4, 32, 32, 32), (33792, 1024, 32, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [output_7, output_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_5.run(buf11, primals_15, primals_16, primals_17, primals_18, primals_19, buf15, 4096, 32, grid=grid(4096, 32), stream=stream0)
        buf13 = reinterpret_tensor(buf16, (4, 1, 32, 32), (33792, 1024, 32, 1), 32768)  # alias
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_6.run(primals_2, buf13, 4096, grid=grid(4096), stream=stream0)
        buf65 = empty_strided_cuda((4, 129, 16, 16), (33024, 1, 2064, 129), torch.float32)
        buf14 = reinterpret_tensor(buf65, (4, 1, 16, 16), (33024, 1, 2064, 129), 128)  # alias
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_7.run(buf13, buf14, 1024, grid=grid(1024), stream=stream0)
        buf17 = empty_strided_cuda((4, 33, 32, 32), (33792, 1, 1056, 33), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf16, buf17, 132, 1024, grid=grid(132, 1024), stream=stream0)
        del buf13
        del buf15
        buf18 = reinterpret_tensor(buf16, (4, 33, 32, 32), (33792, 1, 1056, 33), 0); del buf16  # reuse
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [output_9, output_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_9.run(buf19, buf17, primals_20, primals_21, primals_22, primals_23, primals_24, 135168, grid=grid(135168), stream=stream0)
        # Topologically Sorted Source Nodes: [output_11], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf21 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [output_12, output_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_10.run(buf22, buf20, primals_26, primals_27, primals_28, primals_29, primals_30, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [output_14], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf23, (4, 64, 16, 16), (16384, 1, 1024, 64))
        # Topologically Sorted Source Nodes: [output_15], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf22, primals_32, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf24, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf25 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf26 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [joi_feat, joi_feat_1, joi_feat_2], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_11.run(buf27, buf23, buf24, primals_33, primals_34, primals_35, primals_36, primals_37, buf25, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [output_16], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf29 = empty_strided_cuda((4, 64, 1, 1, 2), (128, 1, 512, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_12.run(buf28, buf29, 512, 128, grid=grid(512), stream=stream0)
        buf30 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf31 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_13.run(buf31, buf29, 256, 2, grid=grid(256), stream=stream0)
        buf32 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf31, (4, 64), (64, 1), 0), reinterpret_tensor(primals_39, (64, 8), (1, 64), 0), out=buf32)
        buf33 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf33, primals_40, 32, grid=grid(32), stream=stream0)
        del primals_40
        buf34 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_42, buf33, reinterpret_tensor(primals_41, (8, 64), (1, 8), 0), alpha=1, beta=1, out=buf34)
        del primals_42
        buf35 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [output_17], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_15.run(buf28, buf34, buf35, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [output_18], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf37 = empty_strided_cuda((4, 32, 16, 16), (8192, 1, 512, 32), torch.float32)
        buf38 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [output_19, output_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_16.run(buf38, buf36, primals_44, primals_45, primals_46, primals_47, primals_48, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_21], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf39, (4, 32, 16, 16), (8192, 1, 512, 32))
        # Topologically Sorted Source Nodes: [output_22], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf38, primals_50, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf40, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf41 = buf23; del buf23  # reuse
        buf42 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_3, output_23], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_17.run(buf39, buf40, primals_51, primals_52, primals_53, primals_54, buf41, buf42, 65536, grid=grid(65536), stream=stream0)
        buf43 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [output_24, adaptive_avg_pool2d_1], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__prelu_kernel_mean_18.run(buf42, primals_55, buf43, 512, 128, grid=grid(512), stream=stream0)
        buf44 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [output_24, adaptive_avg_pool2d_1], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_13.run(buf45, buf43, 256, 2, grid=grid(256), stream=stream0)
        buf46 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf45, (4, 64), (64, 1), 0), reinterpret_tensor(primals_56, (64, 8), (1, 64), 0), out=buf46)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf47, primals_57, 32, grid=grid(32), stream=stream0)
        del primals_57
        buf48 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, buf47, reinterpret_tensor(primals_58, (8, 64), (1, 8), 0), alpha=1, beta=1, out=buf48)
        del primals_59
        buf49 = buf42; del buf42  # reuse
        buf64 = reinterpret_tensor(buf65, (4, 64, 16, 16), (33024, 1, 2064, 129), 64)  # alias
        # Topologically Sorted Source Nodes: [output_24, output_25, output_26, cat_4], Original ATen: [aten._prelu_kernel, aten.mul, aten.add, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_cat_mul_19.run(buf49, buf35, primals_55, buf48, buf64, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [output_27], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_60, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf51 = buf40; del buf40  # reuse
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [output_28, output_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_16.run(buf52, buf50, primals_61, primals_62, primals_63, primals_64, primals_65, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_30], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf53, (4, 32, 16, 16), (8192, 1, 512, 32))
        # Topologically Sorted Source Nodes: [output_31], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf52, primals_67, stride=(1, 1), padding=(2, 2), dilation=(2, 2), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf54, (4, 32, 16, 16), (8192, 1, 512, 32))
        buf55 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        buf56 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_4, output_32], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_17.run(buf53, buf54, primals_68, primals_69, primals_70, primals_71, buf55, buf56, 65536, grid=grid(65536), stream=stream0)
        buf57 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [output_33, adaptive_avg_pool2d_2], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__prelu_kernel_mean_18.run(buf56, primals_72, buf57, 512, 128, grid=grid(512), stream=stream0)
        buf58 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf59 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [output_33, adaptive_avg_pool2d_2], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_13.run(buf59, buf57, 256, 2, grid=grid(256), stream=stream0)
        buf60 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf59, (4, 64), (64, 1), 0), reinterpret_tensor(primals_73, (64, 8), (1, 64), 0), out=buf60)
        buf61 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf61, primals_74, 32, grid=grid(32), stream=stream0)
        del primals_74
        buf62 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_76, buf61, reinterpret_tensor(primals_75, (8, 64), (1, 8), 0), alpha=1, beta=1, out=buf62)
        del primals_76
        buf63 = reinterpret_tensor(buf65, (4, 64, 16, 16), (33024, 1, 2064, 129), 0)  # alias
        # Topologically Sorted Source Nodes: [output_33, output_34, output_35], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_20.run(buf49, buf56, primals_72, buf62, buf63, 65536, grid=grid(65536), stream=stream0)
        buf66 = empty_strided_cuda((4, 129, 16, 16), (33024, 1, 2064, 129), torch.float32)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [output_36, output_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_21.run(buf67, buf65, primals_77, primals_78, primals_79, primals_80, primals_81, 132096, grid=grid(132096), stream=stream0)
        # Topologically Sorted Source Nodes: [output_38], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf69 = reinterpret_tensor(buf54, (4, 128, 8, 8), (8192, 1, 1024, 128), 0); del buf54  # reuse
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [output_39, output_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_22.run(buf70, buf68, primals_83, primals_84, primals_85, primals_86, primals_87, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_41], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf71, (4, 128, 8, 8), (8192, 1, 1024, 128))
        # Topologically Sorted Source Nodes: [output_42], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf70, primals_89, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf72, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf73 = reinterpret_tensor(buf56, (4, 256, 8, 8), (16384, 1, 2048, 256), 0); del buf56  # reuse
        buf74 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_5, joi_feat_6, joi_feat_7], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_23.run(buf75, buf71, buf72, primals_90, primals_91, primals_92, primals_93, primals_94, buf73, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [output_43], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_95, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 128, 8, 8), (8192, 1, 1024, 128))
        buf77 = reinterpret_tensor(buf57, (4, 128, 1, 1), (128, 1, 512, 512), 0); del buf57  # reuse
        buf78 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d_3], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_24.run(buf78, buf76, 512, 64, grid=grid(512), stream=stream0)
        buf79 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf78, (4, 128), (128, 1), 0), reinterpret_tensor(primals_96, (128, 8), (1, 128), 0), out=buf79)
        buf80 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf80, primals_97, 32, grid=grid(32), stream=stream0)
        del primals_97
        buf81 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_99, buf80, reinterpret_tensor(primals_98, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf81)
        del primals_99
        buf82 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [output_44], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_25.run(buf76, buf81, buf82, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_45], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf84 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [output_46, output_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf85, buf83, primals_101, primals_102, primals_103, primals_104, primals_105, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_48], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf86, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_49], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf85, primals_107, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf87, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf88 = buf71; del buf71  # reuse
        buf89 = reinterpret_tensor(buf53, (4, 128, 8, 8), (8192, 1, 1024, 128), 0); del buf53  # reuse
        # Topologically Sorted Source Nodes: [joi_feat_8, output_50], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf86, buf87, primals_108, primals_109, primals_110, primals_111, buf88, buf89, 32768, grid=grid(32768), stream=stream0)
        del buf86
        buf90 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf91 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [output_51, adaptive_avg_pool2d_4], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf91, buf89, primals_112, 512, 64, grid=grid(512), stream=stream0)
        buf92 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf91, (4, 128), (128, 1), 0), reinterpret_tensor(primals_113, (128, 8), (1, 128), 0), out=buf92)
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf93, primals_114, 32, grid=grid(32), stream=stream0)
        del primals_114
        buf94 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_116, buf93, reinterpret_tensor(primals_115, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf94)
        del primals_116
        buf95 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [output_51, output_52, output_53], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf95, buf82, primals_112, buf94, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_54], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf97 = buf87; del buf87  # reuse
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [output_55, output_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf98, buf96, primals_118, primals_119, primals_120, primals_121, primals_122, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_57], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf99, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_58], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf98, primals_124, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf100, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf101 = reinterpret_tensor(buf39, (4, 128, 8, 8), (8192, 1, 1024, 128), 0); del buf39  # reuse
        buf102 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_9, output_59], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf99, buf100, primals_125, primals_126, primals_127, primals_128, buf101, buf102, 32768, grid=grid(32768), stream=stream0)
        del buf100
        buf103 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf104 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [output_60, adaptive_avg_pool2d_5], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf104, buf102, primals_129, 512, 64, grid=grid(512), stream=stream0)
        buf105 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf104, (4, 128), (128, 1), 0), reinterpret_tensor(primals_130, (128, 8), (1, 128), 0), out=buf105)
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf106, primals_131, 32, grid=grid(32), stream=stream0)
        del primals_131
        buf107 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_133, buf106, reinterpret_tensor(primals_132, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf107)
        del primals_133
        buf108 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [output_60, output_61, output_62], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf108, buf95, primals_129, buf107, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_63], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf110 = buf99; del buf99  # reuse
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [output_64, output_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf111, buf109, primals_135, primals_136, primals_137, primals_138, primals_139, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_66], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_140, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf112, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_67], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf111, primals_141, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf113, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf114 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf115 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_10, output_68], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf112, buf113, primals_142, primals_143, primals_144, primals_145, buf114, buf115, 32768, grid=grid(32768), stream=stream0)
        del buf112
        buf116 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [output_69, adaptive_avg_pool2d_6], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf117, buf115, primals_146, 512, 64, grid=grid(512), stream=stream0)
        buf118 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf117, (4, 128), (128, 1), 0), reinterpret_tensor(primals_147, (128, 8), (1, 128), 0), out=buf118)
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf119, primals_148, 32, grid=grid(32), stream=stream0)
        del primals_148
        buf120 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf119, reinterpret_tensor(primals_149, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf120)
        del primals_150
        buf121 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [output_69, output_70, output_71], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf121, buf108, primals_146, buf120, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_72], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf123 = buf113; del buf113  # reuse
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [output_73, output_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf124, buf122, primals_152, primals_153, primals_154, primals_155, primals_156, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_75], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf125, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_76], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf124, primals_158, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf126, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf127 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf128 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_11, output_77], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf125, buf126, primals_159, primals_160, primals_161, primals_162, buf127, buf128, 32768, grid=grid(32768), stream=stream0)
        del buf125
        buf129 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf130 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [output_78, adaptive_avg_pool2d_7], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf130, buf128, primals_163, 512, 64, grid=grid(512), stream=stream0)
        buf131 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf130, (4, 128), (128, 1), 0), reinterpret_tensor(primals_164, (128, 8), (1, 128), 0), out=buf131)
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf132, primals_165, 32, grid=grid(32), stream=stream0)
        del primals_165
        buf133 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_167, buf132, reinterpret_tensor(primals_166, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf133)
        del primals_167
        buf134 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [output_78, output_79, output_80], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf134, buf121, primals_163, buf133, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_81], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf136 = buf126; del buf126  # reuse
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [output_82, output_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf137, buf135, primals_169, primals_170, primals_171, primals_172, primals_173, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_84], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf138, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_85], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf137, primals_175, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf139, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf140 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf141 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_12, output_86], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf138, buf139, primals_176, primals_177, primals_178, primals_179, buf140, buf141, 32768, grid=grid(32768), stream=stream0)
        del buf138
        buf142 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf143 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [output_87, adaptive_avg_pool2d_8], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf143, buf141, primals_180, 512, 64, grid=grid(512), stream=stream0)
        buf144 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf143, (4, 128), (128, 1), 0), reinterpret_tensor(primals_181, (128, 8), (1, 128), 0), out=buf144)
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf145, primals_182, 32, grid=grid(32), stream=stream0)
        del primals_182
        buf146 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_184, buf145, reinterpret_tensor(primals_183, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf146)
        del primals_184
        buf147 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [output_87, output_88, output_89], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf147, buf134, primals_180, buf146, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_90], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_185, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf149 = buf139; del buf139  # reuse
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [output_91, output_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf150, buf148, primals_186, primals_187, primals_188, primals_189, primals_190, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_93], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf151, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_94], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf150, primals_192, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf152, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf153 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf154 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_13, output_95], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf151, buf152, primals_193, primals_194, primals_195, primals_196, buf153, buf154, 32768, grid=grid(32768), stream=stream0)
        del buf151
        buf155 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf156 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [output_96, adaptive_avg_pool2d_9], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf156, buf154, primals_197, 512, 64, grid=grid(512), stream=stream0)
        buf157 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf156, (4, 128), (128, 1), 0), reinterpret_tensor(primals_198, (128, 8), (1, 128), 0), out=buf157)
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf158, primals_199, 32, grid=grid(32), stream=stream0)
        del primals_199
        buf159 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_201, buf158, reinterpret_tensor(primals_200, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf159)
        del primals_201
        buf160 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [output_96, output_97, output_98], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf160, buf147, primals_197, buf159, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_99], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf162 = buf152; del buf152  # reuse
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [output_100, output_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf163, buf161, primals_203, primals_204, primals_205, primals_206, primals_207, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_102], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_208, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf164, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_103], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf163, primals_209, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf165, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf166 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf167 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_14, output_104], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf164, buf165, primals_210, primals_211, primals_212, primals_213, buf166, buf167, 32768, grid=grid(32768), stream=stream0)
        del buf164
        buf168 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf169 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [output_105, adaptive_avg_pool2d_10], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf169, buf167, primals_214, 512, 64, grid=grid(512), stream=stream0)
        buf170 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf169, (4, 128), (128, 1), 0), reinterpret_tensor(primals_215, (128, 8), (1, 128), 0), out=buf170)
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf171, primals_216, 32, grid=grid(32), stream=stream0)
        del primals_216
        buf172 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_218, buf171, reinterpret_tensor(primals_217, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf172)
        del primals_218
        buf173 = buf167; del buf167  # reuse
        # Topologically Sorted Source Nodes: [output_105, output_106, output_107], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf173, buf160, primals_214, buf172, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_108], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_219, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf175 = buf165; del buf165  # reuse
        buf176 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [output_109, output_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf176, buf174, primals_220, primals_221, primals_222, primals_223, primals_224, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_111], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_225, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf177, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_112], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf176, primals_226, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf178, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf179 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf180 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_15, output_113], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf177, buf178, primals_227, primals_228, primals_229, primals_230, buf179, buf180, 32768, grid=grid(32768), stream=stream0)
        del buf177
        buf181 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf182 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [output_114, adaptive_avg_pool2d_11], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf182, buf180, primals_231, 512, 64, grid=grid(512), stream=stream0)
        buf183 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf182, (4, 128), (128, 1), 0), reinterpret_tensor(primals_232, (128, 8), (1, 128), 0), out=buf183)
        buf184 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf184, primals_233, 32, grid=grid(32), stream=stream0)
        del primals_233
        buf185 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_235, buf184, reinterpret_tensor(primals_234, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf185)
        del primals_235
        buf186 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [output_114, output_115, output_116], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf186, buf173, primals_231, buf185, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_117], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf188 = buf178; del buf178  # reuse
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [output_118, output_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf189, buf187, primals_237, primals_238, primals_239, primals_240, primals_241, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_120], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf190, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_121], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf189, primals_243, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf191, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf192 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf193 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_16, output_122], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf190, buf191, primals_244, primals_245, primals_246, primals_247, buf192, buf193, 32768, grid=grid(32768), stream=stream0)
        del buf190
        buf194 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf195 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [output_123, adaptive_avg_pool2d_12], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf195, buf193, primals_248, 512, 64, grid=grid(512), stream=stream0)
        buf196 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf195, (4, 128), (128, 1), 0), reinterpret_tensor(primals_249, (128, 8), (1, 128), 0), out=buf196)
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf197, primals_250, 32, grid=grid(32), stream=stream0)
        del primals_250
        buf198 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_252, buf197, reinterpret_tensor(primals_251, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf198)
        del primals_252
        buf199 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [output_123, output_124, output_125], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf199, buf186, primals_248, buf198, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_126], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf201 = buf191; del buf191  # reuse
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [output_127, output_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf202, buf200, primals_254, primals_255, primals_256, primals_257, primals_258, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_129], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf203, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_130], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf202, primals_260, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf204, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf205 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf206 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_17, output_131], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf203, buf204, primals_261, primals_262, primals_263, primals_264, buf205, buf206, 32768, grid=grid(32768), stream=stream0)
        del buf203
        buf207 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf208 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [output_132, adaptive_avg_pool2d_13], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf208, buf206, primals_265, 512, 64, grid=grid(512), stream=stream0)
        buf209 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf208, (4, 128), (128, 1), 0), reinterpret_tensor(primals_266, (128, 8), (1, 128), 0), out=buf209)
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf210, primals_267, 32, grid=grid(32), stream=stream0)
        del primals_267
        buf211 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_269, buf210, reinterpret_tensor(primals_268, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf211)
        del primals_269
        buf212 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [output_132, output_133, output_134], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf212, buf199, primals_265, buf211, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_135], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf214 = buf204; del buf204  # reuse
        buf215 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [output_136, output_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf215, buf213, primals_271, primals_272, primals_273, primals_274, primals_275, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_138], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf216, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_139], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf215, primals_277, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf217, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf218 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf219 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_18, output_140], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf216, buf217, primals_278, primals_279, primals_280, primals_281, buf218, buf219, 32768, grid=grid(32768), stream=stream0)
        del buf216
        buf220 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf221 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [output_141, adaptive_avg_pool2d_14], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf221, buf219, primals_282, 512, 64, grid=grid(512), stream=stream0)
        buf222 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf221, (4, 128), (128, 1), 0), reinterpret_tensor(primals_283, (128, 8), (1, 128), 0), out=buf222)
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf223, primals_284, 32, grid=grid(32), stream=stream0)
        del primals_284
        buf224 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_286, buf223, reinterpret_tensor(primals_285, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf224)
        del primals_286
        buf225 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [output_141, output_142, output_143], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf225, buf212, primals_282, buf224, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_144], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf227 = buf217; del buf217  # reuse
        buf228 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [output_145, output_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf228, buf226, primals_288, primals_289, primals_290, primals_291, primals_292, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_147], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_293, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf229, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_148], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf228, primals_294, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf230, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf231 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf232 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_19, output_149], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf229, buf230, primals_295, primals_296, primals_297, primals_298, buf231, buf232, 32768, grid=grid(32768), stream=stream0)
        del buf229
        buf233 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf234 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [output_150, adaptive_avg_pool2d_15], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf234, buf232, primals_299, 512, 64, grid=grid(512), stream=stream0)
        buf235 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf234, (4, 128), (128, 1), 0), reinterpret_tensor(primals_300, (128, 8), (1, 128), 0), out=buf235)
        buf236 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf236, primals_301, 32, grid=grid(32), stream=stream0)
        del primals_301
        buf237 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_303, buf236, reinterpret_tensor(primals_302, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf237)
        del primals_303
        buf238 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [output_150, output_151, output_152], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf238, buf225, primals_299, buf237, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_153], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf240 = buf230; del buf230  # reuse
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [output_154, output_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf241, buf239, primals_305, primals_306, primals_307, primals_308, primals_309, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_156], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_310, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf242, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_157], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf241, primals_311, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf243, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf244 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf245 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_20, output_158], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf242, buf243, primals_312, primals_313, primals_314, primals_315, buf244, buf245, 32768, grid=grid(32768), stream=stream0)
        del buf242
        buf246 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf247 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [output_159, adaptive_avg_pool2d_16], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf247, buf245, primals_316, 512, 64, grid=grid(512), stream=stream0)
        buf248 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf247, (4, 128), (128, 1), 0), reinterpret_tensor(primals_317, (128, 8), (1, 128), 0), out=buf248)
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf249, primals_318, 32, grid=grid(32), stream=stream0)
        del primals_318
        buf250 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_320, buf249, reinterpret_tensor(primals_319, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf250)
        del primals_320
        buf251 = buf245; del buf245  # reuse
        # Topologically Sorted Source Nodes: [output_159, output_160, output_161], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf251, buf238, primals_316, buf250, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_162], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf253 = buf243; del buf243  # reuse
        buf254 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [output_163, output_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf254, buf252, primals_322, primals_323, primals_324, primals_325, primals_326, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_165], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf255, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_166], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf254, primals_328, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf256, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf257 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf258 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_21, output_167], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf255, buf256, primals_329, primals_330, primals_331, primals_332, buf257, buf258, 32768, grid=grid(32768), stream=stream0)
        del buf255
        buf259 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf260 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [output_168, adaptive_avg_pool2d_17], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf260, buf258, primals_333, 512, 64, grid=grid(512), stream=stream0)
        buf261 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf260, (4, 128), (128, 1), 0), reinterpret_tensor(primals_334, (128, 8), (1, 128), 0), out=buf261)
        buf262 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [input_72, input_73], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf262, primals_335, 32, grid=grid(32), stream=stream0)
        del primals_335
        buf263 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_337, buf262, reinterpret_tensor(primals_336, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf263)
        del primals_337
        buf264 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [output_168, output_169, output_170], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf264, buf251, primals_333, buf263, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_171], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_338, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf266 = buf256; del buf256  # reuse
        buf267 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [output_172, output_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf267, buf265, primals_339, primals_340, primals_341, primals_342, primals_343, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_174], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf268, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_175], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf267, primals_345, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf269, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf270 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf271 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_22, output_176], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf268, buf269, primals_346, primals_347, primals_348, primals_349, buf270, buf271, 32768, grid=grid(32768), stream=stream0)
        del buf268
        buf272 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf273 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [output_177, adaptive_avg_pool2d_18], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf273, buf271, primals_350, 512, 64, grid=grid(512), stream=stream0)
        buf274 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf273, (4, 128), (128, 1), 0), reinterpret_tensor(primals_351, (128, 8), (1, 128), 0), out=buf274)
        buf275 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf275, primals_352, 32, grid=grid(32), stream=stream0)
        del primals_352
        buf276 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_354, buf275, reinterpret_tensor(primals_353, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf276)
        del primals_354
        buf277 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [output_177, output_178, output_179], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf277, buf264, primals_350, buf276, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_180], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_355, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf279 = buf269; del buf269  # reuse
        buf280 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [output_181, output_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf280, buf278, primals_356, primals_357, primals_358, primals_359, primals_360, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_183], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_361, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf281, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_184], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf280, primals_362, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf282, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf283 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf284 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_23, output_185], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf281, buf282, primals_363, primals_364, primals_365, primals_366, buf283, buf284, 32768, grid=grid(32768), stream=stream0)
        del buf281
        buf285 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf286 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [output_186, adaptive_avg_pool2d_19], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf286, buf284, primals_367, 512, 64, grid=grid(512), stream=stream0)
        buf287 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf286, (4, 128), (128, 1), 0), reinterpret_tensor(primals_368, (128, 8), (1, 128), 0), out=buf287)
        buf288 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf288, primals_369, 32, grid=grid(32), stream=stream0)
        del primals_369
        buf289 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_371, buf288, reinterpret_tensor(primals_370, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf289)
        del primals_371
        buf290 = buf284; del buf284  # reuse
        # Topologically Sorted Source Nodes: [output_186, output_187, output_188], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf290, buf277, primals_367, buf289, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_189], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf292 = buf282; del buf282  # reuse
        buf293 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [output_190, output_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf293, buf291, primals_373, primals_374, primals_375, primals_376, primals_377, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_192], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_378, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf294, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_193], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf293, primals_379, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf295, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf296 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf297 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_24, output_194], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf294, buf295, primals_380, primals_381, primals_382, primals_383, buf296, buf297, 32768, grid=grid(32768), stream=stream0)
        del buf294
        buf298 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf299 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [output_195, adaptive_avg_pool2d_20], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf299, buf297, primals_384, 512, 64, grid=grid(512), stream=stream0)
        buf300 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_84], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf299, (4, 128), (128, 1), 0), reinterpret_tensor(primals_385, (128, 8), (1, 128), 0), out=buf300)
        buf301 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf301, primals_386, 32, grid=grid(32), stream=stream0)
        del primals_386
        buf302 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_388, buf301, reinterpret_tensor(primals_387, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf302)
        del primals_388
        buf303 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [output_195, output_196, output_197], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf303, buf290, primals_384, buf302, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_198], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_389, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf305 = buf295; del buf295  # reuse
        buf306 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [output_199, output_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf306, buf304, primals_390, primals_391, primals_392, primals_393, primals_394, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_201], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_395, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf307, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_202], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf306, primals_396, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf308, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf309 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf310 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_25, output_203], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf307, buf308, primals_397, primals_398, primals_399, primals_400, buf309, buf310, 32768, grid=grid(32768), stream=stream0)
        del buf307
        buf311 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf312 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [output_204, adaptive_avg_pool2d_21], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf312, buf310, primals_401, 512, 64, grid=grid(512), stream=stream0)
        buf313 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf312, (4, 128), (128, 1), 0), reinterpret_tensor(primals_402, (128, 8), (1, 128), 0), out=buf313)
        buf314 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [input_88, input_89], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf314, primals_403, 32, grid=grid(32), stream=stream0)
        del primals_403
        buf315 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_90], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_405, buf314, reinterpret_tensor(primals_404, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf315)
        del primals_405
        buf316 = buf310; del buf310  # reuse
        # Topologically Sorted Source Nodes: [output_204, output_205, output_206], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf316, buf303, primals_401, buf315, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_207], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_406, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf318 = buf308; del buf308  # reuse
        buf319 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [output_208, output_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf319, buf317, primals_407, primals_408, primals_409, primals_410, primals_411, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_210], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf320, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_211], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf319, primals_413, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf321, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf322 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf323 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_26, output_212], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf320, buf321, primals_414, primals_415, primals_416, primals_417, buf322, buf323, 32768, grid=grid(32768), stream=stream0)
        del buf320
        buf324 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf325 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [output_213, adaptive_avg_pool2d_22], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf325, buf323, primals_418, 512, 64, grid=grid(512), stream=stream0)
        buf326 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_92], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf325, (4, 128), (128, 1), 0), reinterpret_tensor(primals_419, (128, 8), (1, 128), 0), out=buf326)
        buf327 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf327, primals_420, 32, grid=grid(32), stream=stream0)
        del primals_420
        buf328 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_422, buf327, reinterpret_tensor(primals_421, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf328)
        del primals_422
        buf329 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [output_213, output_214, output_215], Original ATen: [aten._prelu_kernel, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__prelu_kernel_add_mul_29.run(buf329, buf316, primals_418, buf328, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [output_216], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_423, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf331 = buf321; del buf321  # reuse
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [output_217, output_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_26.run(buf332, buf330, primals_424, primals_425, primals_426, primals_427, primals_428, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [output_219], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_429, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf333, (4, 64, 8, 8), (4096, 1, 512, 64))
        # Topologically Sorted Source Nodes: [output_220], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf332, primals_430, stride=(1, 1), padding=(4, 4), dilation=(4, 4), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf334, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf335 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        buf336 = empty_strided_cuda((4, 128, 8, 8), (8192, 1, 1024, 128), torch.float32)
        # Topologically Sorted Source Nodes: [joi_feat_27, output_221], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_cat_27.run(buf333, buf334, primals_431, primals_432, primals_433, primals_434, buf335, buf336, 32768, grid=grid(32768), stream=stream0)
        del buf333
        del buf334
        buf337 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 512, 512), torch.float32)
        buf338 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [output_222, adaptive_avg_pool2d_23], Original ATen: [aten._prelu_kernel, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__prelu_kernel_mean_28.run(buf338, buf336, primals_435, 512, 64, grid=grid(512), stream=stream0)
        buf339 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf338, (4, 128), (128, 1), 0), reinterpret_tensor(primals_436, (128, 8), (1, 128), 0), out=buf339)
        buf340 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [input_96, input_97], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_14.run(buf340, primals_437, 32, grid=grid(32), stream=stream0)
        del primals_437
        buf341 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_439, buf340, reinterpret_tensor(primals_438, (8, 128), (1, 8), 0), alpha=1, beta=1, out=buf341)
        del primals_439
        buf342 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf343 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf344 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [cat_26, output_225, output_226], Original ATen: [aten.cat, aten._native_batch_norm_legit_no_training, aten._prelu_kernel]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training__prelu_kernel_cat_30.run(buf344, buf82, buf329, buf336, primals_435, buf341, primals_440, primals_441, primals_442, primals_443, primals_444, buf342, 65536, grid=grid(65536), stream=stream0)
        del buf336
        # Topologically Sorted Source Nodes: [output_227], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, primals_445, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 19, 8, 8), (1216, 1, 152, 19))
        buf346 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf346, 64, grid=grid(64), stream=stream0)
        buf347 = empty_strided_cuda((64, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_32.run(buf347, 64, grid=grid(64), stream=stream0)
        buf348 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_31.run(buf348, 64, grid=grid(64), stream=stream0)
        buf349 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_32.run(buf349, 64, grid=grid(64), stream=stream0)
        buf350 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_33.run(buf350, 64, grid=grid(64), stream=stream0)
        buf352 = empty_strided_cuda((64, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_mul_sub_33.run(buf352, 64, grid=grid(64), stream=stream0)
        buf351 = empty_strided_cuda((4, 19, 64, 64), (77824, 4096, 64, 1), torch.float32)
        buf353 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_34.run(buf353, buf346, buf348, buf345, buf349, buf350, buf347, buf352, 311296, grid=grid(311296), stream=stream0)
        del buf345
    return (buf353, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, buf0, primals_9, primals_10, primals_11, primals_12, primals_13, buf1, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, buf2, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_77, primals_78, primals_79, primals_80, primals_81, buf3, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, buf5, buf7, buf8, buf10, buf11, buf17, buf19, buf20, buf22, buf25, buf27, buf28, reinterpret_tensor(buf31, (4, 64), (64, 1), 0), buf33, buf34, buf35, buf36, buf38, buf41, reinterpret_tensor(buf45, (4, 64), (64, 1), 0), buf47, buf48, buf49, buf50, buf52, buf55, reinterpret_tensor(buf59, (4, 64), (64, 1), 0), buf61, buf62, buf65, buf67, buf68, buf70, buf73, buf75, buf76, reinterpret_tensor(buf78, (4, 128), (128, 1), 0), buf80, buf81, buf82, buf83, buf85, buf88, reinterpret_tensor(buf91, (4, 128), (128, 1), 0), buf93, buf94, buf95, buf96, buf98, buf101, reinterpret_tensor(buf104, (4, 128), (128, 1), 0), buf106, buf107, buf108, buf109, buf111, buf114, reinterpret_tensor(buf117, (4, 128), (128, 1), 0), buf119, buf120, buf121, buf122, buf124, buf127, reinterpret_tensor(buf130, (4, 128), (128, 1), 0), buf132, buf133, buf134, buf135, buf137, buf140, reinterpret_tensor(buf143, (4, 128), (128, 1), 0), buf145, buf146, buf147, buf148, buf150, buf153, reinterpret_tensor(buf156, (4, 128), (128, 1), 0), buf158, buf159, buf160, buf161, buf163, buf166, reinterpret_tensor(buf169, (4, 128), (128, 1), 0), buf171, buf172, buf173, buf174, buf176, buf179, reinterpret_tensor(buf182, (4, 128), (128, 1), 0), buf184, buf185, buf186, buf187, buf189, buf192, reinterpret_tensor(buf195, (4, 128), (128, 1), 0), buf197, buf198, buf199, buf200, buf202, buf205, reinterpret_tensor(buf208, (4, 128), (128, 1), 0), buf210, buf211, buf212, buf213, buf215, buf218, reinterpret_tensor(buf221, (4, 128), (128, 1), 0), buf223, buf224, buf225, buf226, buf228, buf231, reinterpret_tensor(buf234, (4, 128), (128, 1), 0), buf236, buf237, buf238, buf239, buf241, buf244, reinterpret_tensor(buf247, (4, 128), (128, 1), 0), buf249, buf250, buf251, buf252, buf254, buf257, reinterpret_tensor(buf260, (4, 128), (128, 1), 0), buf262, buf263, buf264, buf265, buf267, buf270, reinterpret_tensor(buf273, (4, 128), (128, 1), 0), buf275, buf276, buf277, buf278, buf280, buf283, reinterpret_tensor(buf286, (4, 128), (128, 1), 0), buf288, buf289, buf290, buf291, buf293, buf296, reinterpret_tensor(buf299, (4, 128), (128, 1), 0), buf301, buf302, buf303, buf304, buf306, buf309, reinterpret_tensor(buf312, (4, 128), (128, 1), 0), buf314, buf315, buf316, buf317, buf319, buf322, reinterpret_tensor(buf325, (4, 128), (128, 1), 0), buf327, buf328, buf329, buf330, buf332, buf335, reinterpret_tensor(buf338, (4, 128), (128, 1), 0), buf340, buf341, buf342, buf344, buf346, buf347, buf348, buf349, buf350, buf352, primals_438, primals_436, primals_421, primals_419, primals_404, primals_402, primals_387, primals_385, primals_370, primals_368, primals_353, primals_351, primals_336, primals_334, primals_319, primals_317, primals_302, primals_300, primals_285, primals_283, primals_268, primals_266, primals_251, primals_249, primals_234, primals_232, primals_217, primals_215, primals_200, primals_198, primals_183, primals_181, primals_166, primals_164, primals_149, primals_147, primals_132, primals_130, primals_115, primals_113, primals_98, primals_96, primals_75, primals_73, primals_58, primals_56, primals_41, primals_39, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 1, 64, 64), (4096, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((33, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((33, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((33, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((33, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((33, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, 33, 3, 3), (297, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((8, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((64, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((8, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((8, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((129, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((129, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((129, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((129, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((129, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 129, 3, 3), (1161, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((8, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((128, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((19, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
