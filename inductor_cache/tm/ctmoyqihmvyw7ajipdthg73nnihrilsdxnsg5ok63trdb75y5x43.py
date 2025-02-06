# AOT ID: ['20_forward']
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


# kernel path: inductor_cache/fn/cfnmi3ksr4k2fsmkxwjga32o74myz2bujyhuq2i47qgzaisawhum.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => _adaptive_avg_pool2d
# Graph fragment:
#   %_adaptive_avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%primals_1, [24, 1]), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_0 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24)
    x1 = xindex // 24
    x2 = xindex
    tmp0 = x0 // 6
    tmp1 = (27 + 4*x0) // 24
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.full([1], 4, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (4*(x0 // 6) + 16*x1), tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = tl.full([1], 1, tl.int64)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 4*(x0 // 6) + 16*x1), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tl.full([1], 2, tl.int64)
    tmp14 = tmp13 < tmp4
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + (2 + 4*(x0 // 6) + 16*x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tl.full([1], 3, tl.int64)
    tmp19 = tmp18 < tmp4
    tmp20 = tmp2 & tmp19
    tmp21 = tl.load(in_ptr0 + (3 + 4*(x0 // 6) + 16*x1), tmp20, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = 1 + (x0 // 6)
    tmp24 = tmp23 < tmp1
    tmp25 = tmp24 & tmp5
    tmp26 = tl.load(in_ptr0 + (4 + 4*(x0 // 6) + 16*x1), tmp25, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26 + tmp22
    tmp28 = tmp24 & tmp9
    tmp29 = tl.load(in_ptr0 + (5 + 4*(x0 // 6) + 16*x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp27
    tmp31 = tmp24 & tmp14
    tmp32 = tl.load(in_ptr0 + (6 + 4*(x0 // 6) + 16*x1), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp30
    tmp34 = tmp24 & tmp19
    tmp35 = tl.load(in_ptr0 + (7 + 4*(x0 // 6) + 16*x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = 1.0
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp6, tmp37, tmp38)
    tmp40 = 1.0
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp10, tmp40, tmp41)
    tmp43 = tmp42 + tmp39
    tmp44 = 1.0
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp15, tmp44, tmp45)
    tmp47 = tmp46 + tmp43
    tmp48 = 1.0
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp20, tmp48, tmp49)
    tmp51 = tmp50 + tmp47
    tmp52 = 1.0
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp25, tmp52, tmp53)
    tmp55 = tmp54 + tmp51
    tmp56 = 1.0
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp28, tmp56, tmp57)
    tmp59 = tmp58 + tmp55
    tmp60 = 1.0
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp31, tmp60, tmp61)
    tmp63 = tmp62 + tmp59
    tmp64 = 1.0
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp34, tmp64, tmp65)
    tmp67 = tmp66 + tmp63
    tmp68 = tmp36 / tmp67
    tl.store(out_ptr0 + (x2), tmp68, None)
''', device_str='cuda')


# kernel path: inductor_cache/3w/c3we22s5htv3svkwiigl2z7t7czlyoo5cz65wcqj276jffvcqrhp.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d_1], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d_1 => _adaptive_avg_pool2d_1
# Graph fragment:
#   %_adaptive_avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%primals_1, [1, 24]), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_1 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24)
    x1 = xindex // 24
    x2 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0 // 6
    tmp4 = (27 + 4*x0) // 24
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (16*x1 + (x0 // 6)), tmp6, eviction_policy='evict_last', other=0.0)
    tmp8 = 1 + (x0 // 6)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 16*x1 + (x0 // 6)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tl.full([1], 1, tl.int64)
    tmp14 = tmp13 < tmp1
    tmp15 = tmp14 & tmp5
    tmp16 = tl.load(in_ptr0 + (4 + 16*x1 + (x0 // 6)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp14 & tmp9
    tmp19 = tl.load(in_ptr0 + (5 + 16*x1 + (x0 // 6)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 + tmp17
    tmp21 = tl.full([1], 2, tl.int64)
    tmp22 = tmp21 < tmp1
    tmp23 = tmp22 & tmp5
    tmp24 = tl.load(in_ptr0 + (8 + 16*x1 + (x0 // 6)), tmp23, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp20
    tmp26 = tmp22 & tmp9
    tmp27 = tl.load(in_ptr0 + (9 + 16*x1 + (x0 // 6)), tmp26, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp27 + tmp25
    tmp29 = tl.full([1], 3, tl.int64)
    tmp30 = tmp29 < tmp1
    tmp31 = tmp30 & tmp5
    tmp32 = tl.load(in_ptr0 + (12 + 16*x1 + (x0 // 6)), tmp31, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32 + tmp28
    tmp34 = tmp30 & tmp9
    tmp35 = tl.load(in_ptr0 + (13 + 16*x1 + (x0 // 6)), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp35 + tmp33
    tmp37 = 1.0
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp6, tmp37, tmp38)
    tmp40 = 1.0
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp10, tmp40, tmp41)
    tmp43 = tmp42 + tmp39
    tmp44 = 1.0
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp15, tmp44, tmp45)
    tmp47 = tmp46 + tmp43
    tmp48 = 1.0
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp18, tmp48, tmp49)
    tmp51 = tmp50 + tmp47
    tmp52 = 1.0
    tmp53 = tl.full(tmp52.shape, 0.0, tmp52.dtype)
    tmp54 = tl.where(tmp23, tmp52, tmp53)
    tmp55 = tmp54 + tmp51
    tmp56 = 1.0
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp26, tmp56, tmp57)
    tmp59 = tmp58 + tmp55
    tmp60 = 1.0
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp31, tmp60, tmp61)
    tmp63 = tmp62 + tmp59
    tmp64 = 1.0
    tmp65 = tl.full(tmp64.shape, 0.0, tmp64.dtype)
    tmp66 = tl.where(tmp34, tmp64, tmp65)
    tmp67 = tmp66 + tmp63
    tmp68 = tmp36 / tmp67
    tl.store(out_ptr0 + (x2), tmp68, None)
''', device_str='cuda')


# kernel path: inductor_cache/6e/c6e4xc5alaevtymgzub47tzpdonfonwysna2m5sfvw77iltfwaqr.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 24) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/4y/c4y6wkb6iuvoaqw2zsaufmjbbglaed5jstiw24f3ca3ojt7v2wyt.py
# Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.sigmoid]
# Source node to ATen node mapping:
#   input_7 => convolution_2
#   input_8 => sigmoid
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_12, %primals_13, [1], [1], [1], False, [0], 1), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_2,), kwargs = {})
triton_poi_fused_convolution_sigmoid_3 = async_compile.triton('triton_poi_fused_convolution_sigmoid_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_sigmoid_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_sigmoid_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 24) % 20)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vn/cvnvxcoif33df7u6jzxashheuwa7zexxhkctvqikczu4uqzw427l.py
# Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_11 => convolution_4
# Graph fragment:
#   %convolution_4 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_16, %primals_17, [1], [3], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_4 = async_compile.triton('triton_poi_fused_convolution_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 24) % 512)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/bx/cbxivm675cqqw6ngrtlo73a4lr7da5wo7gfiffd4cxut6a4btuhc.py
# Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   fea_hup_1 => convert_element_type_5
# Graph fragment:
#   %convert_element_type_5 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 7.666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/el/cel2f4kjckyvpsccoqb2rx4aoy7v5lyzocetyt2r3yrk6z6hj2br.py
# Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   fea_hup_1 => add_4, clamp_max
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_5, 1), kwargs = {})
#   %clamp_max : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_4, 23), kwargs = {})
triton_poi_fused_add_clamp_6 = async_compile.triton('triton_poi_fused_add_clamp_6', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_6(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 7.666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 23, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7b/c7b23zltcsb74v7sqeyuvmwdnewkyud3qp36wx3zq6gmbgj4k6ek.py
# Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
# Source node to ATen node mapping:
#   fea_hup_1 => clamp_min_1, convert_element_type_4, convert_element_type_7, iota, mul_7
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %clamp_min_1 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_7, 0.0), kwargs = {})
#   %convert_element_type_7 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_7 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_7', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int64)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pf/cpf2d3uaiyyrxse72nsdld2kuuaxfimt6yn6osuaddtmrnnibixi.py
# Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   fea_hup_1 => clamp_max_2, clamp_min_1, clamp_min_2, convert_element_type_4, iota, mul_7, sub_2
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, 0.0), kwargs = {})
#   %clamp_min_1 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_7, 0.0), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_7), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_8 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_8', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_8(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tq/ctqkc4nwk2jnsfpphqwtouiyxhpsqmwnyyztjwrsyuig6zrs7upt.py
# Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   fea_hup_1 => clamp_max_3, clamp_min_3, sub_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_5), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
triton_poi_fused_clamp_sub_9 = async_compile.triton('triton_poi_fused_clamp_sub_9', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_sub_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_sub_9(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 7.666666666666667
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4y/c4yduovk56ibc3dqb3pdkyagytufib2ceaq7q6qpazcy7ne6v5co.py
# Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   fea_hup_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_6, add_7, add_8, mul_10, mul_8, mul_9, sub_3, sub_4, sub_6
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_8, [None, None, %convert_element_type_5, %convert_element_type_7]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_8, [None, None, %convert_element_type_5, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_8, [None, None, %clamp_max, %convert_element_type_7]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_8, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %clamp_max_2), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_8), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_2), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_9), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %add_6), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_3), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_10), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_10 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 24, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tl.full([XBLOCK], 1, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp10 = tl.load(in_ptr2 + (tmp4 + 24*x2), None, eviction_policy='evict_last')
    tmp11 = tl.sigmoid(tmp10)
    tmp13 = tmp12 + tmp6
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tmp16 = tmp11 - tmp11
    tmp18 = tmp16 * tmp17
    tmp19 = tmp11 + tmp18
    tmp21 = tmp20 + tmp1
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr2 + (tmp23 + 24*x2), None, eviction_policy='evict_last')
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp25 - tmp25
    tmp27 = tmp26 * tmp17
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 - tmp19
    tmp31 = tmp29 * tmp30
    tmp32 = tmp19 + tmp31
    tl.store(in_out_ptr0 + (x4), tmp32, None)
''', device_str='cuda')


# kernel path: inductor_cache/df/cdfuqwffxkppabplmuvo3ogt4y7u3sgwy6wnnjq7k2uw2g57vtlq.py
# Topologically Sorted Source Nodes: [fea_wup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   fea_wup_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_11, add_12, add_13, mul_13, mul_14, mul_15, sub_11, sub_8, sub_9
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_9, [None, None, %convert_element_type_9, %convert_element_type_11]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_9, [None, None, %convert_element_type_9, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_9, [None, None, %clamp_max_4, %convert_element_type_11]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%unsqueeze_9, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %clamp_max_6), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_13), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %clamp_max_6), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_14), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_12, %add_11), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %clamp_max_7), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %mul_15), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_11 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tl.full([XBLOCK], 24, tl.int32)
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tmp10 = tl.load(in_ptr2 + (tmp9 + 24*x2), None, eviction_policy='evict_last')
    tmp11 = tl.sigmoid(tmp10)
    tmp13 = tmp12 + tmp6
    tmp14 = tmp12 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp12)
    tmp16 = tl.load(in_ptr2 + (tmp15 + 24*x2), None, eviction_policy='evict_last')
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp17 - tmp11
    tmp20 = tmp18 * tmp19
    tmp21 = tmp11 + tmp20
    tmp23 = tmp22 + tmp1
    tmp24 = tmp22 < 0
    tmp25 = tl.where(tmp24, tmp23, tmp22)
    tmp26 = tmp21 - tmp21
    tmp28 = tmp26 * tmp27
    tmp29 = tmp21 + tmp28
    tl.store(in_out_ptr0 + (x4), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/5e/c5eko4gy2xzjknw3snhjp5vyexe2eebjgpcarz5mlus7jf4njllu.py
# Topologically Sorted Source Nodes: [fea], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   fea => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_1, %mul_18, %add_14], 1), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 1536)
    x0 = (xindex % 16)
    x2 = xindex // 24576
    x3 = xindex
    tmp11 = tl.load(in_ptr1 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp15 = tl.load(in_ptr3 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (0))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp30 = tl.load(in_ptr3 + (0))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 16*(x1) + 8192*x2), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1024, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (x0 + 16*((-512) + x1) + 8192*x2), tmp9, other=0.0)
    tmp13 = tl.load(in_ptr2 + (x0 + 16*((-512) + x1) + 8192*x2), tmp9, other=0.0)
    tmp14 = tmp12 * tmp13
    tmp17 = tl.load(in_ptr4 + (x0 + 16*((-512) + x1) + 8192*x2), tmp9, other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tmp14 + tmp18
    tmp20 = tmp10 * tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp9, tmp20, tmp21)
    tmp23 = tmp0 >= tmp7
    tmp24 = tl.full([1], 1536, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp28 = tl.load(in_ptr2 + (x0 + 16*((-1024) + x1) + 8192*x2), tmp23, other=0.0)
    tmp29 = tmp27 * tmp28
    tmp32 = tl.load(in_ptr4 + (x0 + 16*((-1024) + x1) + 8192*x2), tmp23, other=0.0)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp29 + tmp33
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp23, tmp34, tmp35)
    tmp37 = tl.where(tmp9, tmp22, tmp36)
    tmp38 = tl.where(tmp4, tmp5, tmp37)
    tl.store(out_ptr0 + (x3), tmp38, None)
''', device_str='cuda')


# kernel path: inductor_cache/o3/co3lffz24mhzmrshakxnv2vadofj243bxvssmdfhrxjmih26mugz.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_16 => add_16, mul_20, mul_21, sub_12
#   input_17 => relu_2
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_11), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_13), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_15), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_17), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_2, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tl.store(out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26 = args
    args.clear()
    assert_size_stride(primals_1, (4, 512, 4, 4), (8192, 16, 4, 1))
    assert_size_stride(primals_2, (256, 512, 3), (1536, 3, 1))
    assert_size_stride(primals_3, (256, ), (1, ))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (256, 512, 3), (1536, 3, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (256, ), (1, ))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (20, 256, 3), (768, 3, 1))
    assert_size_stride(primals_13, (20, ), (1, ))
    assert_size_stride(primals_14, (20, 256, 3), (768, 3, 1))
    assert_size_stride(primals_15, (20, ), (1, ))
    assert_size_stride(primals_16, (512, 256, 7), (1792, 7, 1))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, 256, 7), (1792, 7, 1))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (1, ), (1, ))
    assert_size_stride(primals_21, (1, ), (1, ))
    assert_size_stride(primals_22, (512, 1536, 3, 3), (13824, 9, 3, 1))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 512, 24, 1), (12288, 24, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten._adaptive_avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_0.run(primals_1, buf0, 49152, grid=grid(49152), stream=stream0)
        buf1 = empty_strided_cuda((4, 512, 1, 24), (12288, 24, 24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d_1], Original ATen: [aten._adaptive_avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1.run(primals_1, buf1, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(reinterpret_tensor(buf0, (4, 512, 24), (12288, 24, 1), 0), primals_2, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf2, (4, 256, 24), (6144, 24, 1))
        buf3 = empty_strided_cuda((4, 256, 24), (6144, 24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, primals_3, primals_4, primals_5, primals_6, buf3, 24576, grid=grid(24576), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(reinterpret_tensor(buf1, (4, 512, 24), (12288, 24, 1), 0), primals_7, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf4, (4, 256, 24), (6144, 24, 1))
        buf5 = empty_strided_cuda((4, 256, 24), (6144, 24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, primals_8, primals_9, primals_10, primals_11, buf5, 24576, grid=grid(24576), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf3, primals_12, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf6, (4, 20, 24), (480, 24, 1))
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [input_7, input_8], Original ATen: [aten.convolution, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_sigmoid_3.run(buf7, primals_13, 1920, grid=grid(1920), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf5, primals_14, stride=(1,), padding=(1,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf8, (4, 20, 24), (480, 24, 1))
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.convolution, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_sigmoid_3.run(buf9, primals_15, 1920, grid=grid(1920), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf3, primals_16, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf10, (4, 512, 24), (12288, 24, 1))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_4.run(buf11, primals_17, 49152, grid=grid(49152), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf5, primals_18, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf12, (4, 512, 24), (12288, 24, 1))
        buf13 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_4.run(buf13, primals_19, 49152, grid=grid(49152), stream=stream0)
        del primals_19
        buf14 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf14, 4, grid=grid(4), stream=stream0)
        buf15 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_6.run(buf15, 4, grid=grid(4), stream=stream0)
        buf16 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_7.run(buf16, 4, grid=grid(4), stream=stream0)
        buf17 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_7.run(buf17, 4, grid=grid(4), stream=stream0)
        buf18 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_8.run(buf18, 4, grid=grid(4), stream=stream0)
        buf20 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_sub_9.run(buf20, 4, grid=grid(4), stream=stream0)
        buf19 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf21 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [fea_hup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_10.run(buf21, buf14, buf16, buf11, buf17, buf18, buf15, buf20, 32768, grid=grid(32768), stream=stream0)
        buf22 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [fea_wup_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_7.run(buf22, 4, grid=grid(4), stream=stream0)
        buf23 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [fea_wup_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_7.run(buf23, 4, grid=grid(4), stream=stream0)
        buf24 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [fea_hup_1, fea_wup_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf24, 4, grid=grid(4), stream=stream0)
        buf25 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [fea_wup_1], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_6.run(buf25, 4, grid=grid(4), stream=stream0)
        buf26 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [fea_hup_1, fea_wup_1], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_sub_9.run(buf26, 4, grid=grid(4), stream=stream0)
        buf28 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [fea_wup_1], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_8.run(buf28, 4, grid=grid(4), stream=stream0)
        buf27 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf29 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [fea_wup_1], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_11.run(buf29, buf22, buf24, buf13, buf25, buf26, buf23, buf28, 32768, grid=grid(32768), stream=stream0)
        buf30 = empty_strided_cuda((4, 1536, 4, 4), (24576, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [fea], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(primals_1, primals_20, buf29, primals_21, buf21, buf30, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf32 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        buf33 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13.run(buf31, primals_23, primals_24, primals_25, primals_26, buf32, buf33, 32768, grid=grid(32768), stream=stream0)
        del primals_26
    return (buf32, buf7, buf9, primals_1, primals_2, primals_3, primals_4, primals_5, primals_7, primals_8, primals_9, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, reinterpret_tensor(buf0, (4, 512, 24), (12288, 24, 1), 0), reinterpret_tensor(buf1, (4, 512, 24), (12288, 24, 1), 0), buf2, buf3, buf4, buf5, buf7, buf9, buf11, buf13, buf14, buf15, buf16, buf17, buf18, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf28, buf29, buf30, buf31, buf33, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 512, 4, 4), (8192, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((256, 512, 3), (1536, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 512, 3), (1536, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((20, 256, 3), (768, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((20, 256, 3), (768, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, 256, 7), (1792, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, 256, 7), (1792, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, 1536, 3, 3), (13824, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
