# AOT ID: ['9_inference']
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


# kernel path: inductor_cache/ig/cigfsdbhdadikhcbkyumtxv62udg56utlsx72h4xgb3uxfcggou4.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, 1), kwargs = {})
#   %amax_default_3 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_3, [-1], True), kwargs = {})
#   %sub_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_3, %amax_default_3), kwargs = {})
#   %div_tensor_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_3, 0.1), kwargs = {})
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 10.0
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3c/c3cvutfgxjjuxbotpufx5syvzbzicc66binfft7mjvi64ccttgiv.py
# Topologically Sorted Source Nodes: [lsm], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   lsm => exp, log, sub_1, sum_1
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_3,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_1,), kwargs = {})
#   %sub_1 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_tensor_3, %log), kwargs = {})
triton_poi_fused__log_softmax_1 = async_compile.triton('triton_poi_fused__log_softmax_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl_math.exp(tmp1)
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp2 + tmp4
    tmp7 = tl_math.exp(tmp6)
    tmp8 = tmp5 + tmp7
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp0 - tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wz/cwzllsrrg3iurvcjimnzb7tx53cfdmxsfcxo73jfuvbqquxqwaap.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 0.1), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (128 + x2), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + 4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (129 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (130 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (131 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 10.0
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pi/cpindct3j4vdyrfzuvbppte2iywlmcytced7uzyj67xybe5v5lj3.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 0.1), kwargs = {})
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (192 + x2), xmask)
    tmp3 = tl.load(in_ptr0 + (192 + 4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (193 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (194 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (195 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 10.0
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ne/cnebjqkaarf3c54cxorkvlat7cgoeskgwpt4o2gmlb76kwcmxjm2.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mul_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, 1), kwargs = {})
#   %amax_default_2 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_2, [-1], True), kwargs = {})
#   %sub_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_2, %amax_default_2), kwargs = {})
#   %div_tensor_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_2, 0.1), kwargs = {})
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (64 + x2), xmask)
    tmp3 = tl.load(in_ptr0 + (64 + 4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (66 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (67 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 10.0
    tmp16 = tmp14 * tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4h/c4ho4wx3vaztydywqmxzmgfz5igziygcxqc4ujb644eghuvecl2c.py
# Topologically Sorted Source Nodes: [mul, loss, mean, total_loss, mul_1, loss_1, mean_1, total_loss_1, mul_2, loss_2, mean_2, total_loss_2, mul_3, loss_3, mean_3, total_loss_3, mul_4, loss_4, mean_4, total_loss_4, mul_5, loss_5, mean_5, total_loss_5, mul_6, loss_6, mean_6, total_loss_6, mul_7, loss_7, mean_7, total_loss_7, mul_8, loss_8, mean_8, total_loss_8, mul_9, loss_9, mean_9, total_loss_9, mul_10, loss_10, mean_10, total_loss_10, mul_11, loss_11, mean_11, total_loss_11, mul_12, loss_12, mean_12, total_loss_12, mul_13, loss_13, mean_13, total_loss_13, mul_14, loss_14, mean_14, total_loss_14, mul_15, loss_15, mean_15, total_loss_15], Original ATen: [aten.mul, aten.sum, aten.mean, aten.rsub, aten.sub]
# Source node to ATen node mapping:
#   loss => sum_2
#   loss_1 => sum_3
#   loss_10 => sum_14
#   loss_11 => sum_15
#   loss_12 => sum_17
#   loss_13 => sum_18
#   loss_14 => sum_19
#   loss_15 => sum_20
#   loss_2 => sum_4
#   loss_3 => sum_5
#   loss_4 => sum_7
#   loss_5 => sum_8
#   loss_6 => sum_9
#   loss_7 => sum_10
#   loss_8 => sum_12
#   loss_9 => sum_13
#   mean => mean
#   mean_1 => mean_1
#   mean_10 => mean_10
#   mean_11 => mean_11
#   mean_12 => mean_12
#   mean_13 => mean_13
#   mean_14 => mean_14
#   mean_15 => mean_15
#   mean_2 => mean_2
#   mean_3 => mean_3
#   mean_4 => mean_4
#   mean_5 => mean_5
#   mean_6 => mean_6
#   mean_7 => mean_7
#   mean_8 => mean_8
#   mean_9 => mean_9
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   total_loss => sub_2
#   total_loss_1 => sub_3
#   total_loss_10 => sub_16
#   total_loss_11 => sub_17
#   total_loss_12 => sub_20
#   total_loss_13 => sub_21
#   total_loss_14 => sub_22
#   total_loss_15 => sub_23
#   total_loss_2 => sub_4
#   total_loss_3 => sub_5
#   total_loss_4 => sub_8
#   total_loss_5 => sub_9
#   total_loss_6 => sub_10
#   total_loss_7 => sub_11
#   total_loss_8 => sub_14
#   total_loss_9 => sub_15
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %sub_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [-1]), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_2,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (0, %mean), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %sub_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [-1]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_3,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_2, %mean_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_6, %sub_1), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [-1]), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_4,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %mean_2), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %sub_1), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [-1]), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_5,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_4, %mean_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %sub_7), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_4, [-1]), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_7,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_5, %mean_4), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_9, %sub_7), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [-1]), kwargs = {})
#   %mean_5 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_8,), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_8, %mean_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %sub_7), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_6, [-1]), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_9,), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_9, %mean_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_11, %sub_7), kwargs = {})
#   %sum_10 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [-1]), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_10,), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_10, %mean_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_12, %sub_13), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [-1]), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_12,), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_11, %mean_8), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %sub_13), kwargs = {})
#   %sum_13 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_9, [-1]), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_13,), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_14, %mean_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_14, %sub_13), kwargs = {})
#   %sum_14 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_10, [-1]), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_14,), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_15, %mean_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %sub_13), kwargs = {})
#   %sum_15 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_11, [-1]), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_15,), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_16, %mean_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %sub_19), kwargs = {})
#   %sum_17 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12, [-1]), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_17,), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_17, %mean_12), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_17, %sub_19), kwargs = {})
#   %sum_18 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13, [-1]), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_18,), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_20, %mean_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %sub_19), kwargs = {})
#   %sum_19 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_14, [-1]), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_19,), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_21, %mean_14), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %sub_19), kwargs = {})
#   %sum_20 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_15, [-1]), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%sum_20,), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_22, %mean_15), kwargs = {})
triton_per_fused_mean_mul_rsub_sub_sum_5 = async_compile.triton('triton_per_fused_mean_mul_rsub_sub_sum_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_rsub_sub_sum_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 16, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_rsub_sub_sum_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (64 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (65 + 4*r0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (66 + 4*r0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (67 + 4*r0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (128 + 4*r0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (129 + 4*r0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (130 + 4*r0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (131 + 4*r0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (192 + 4*r0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (193 + 4*r0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (194 + 4*r0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (195 + 4*r0), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr3 + (4*r0), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr3 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr3 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr3 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp148 = tl.load(in_ptr4 + (4*r0), None, eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr4 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr4 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr4 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp19 = tmp18 * tmp1
    tmp21 = tmp20 * tmp4
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23 * tmp8
    tmp25 = tmp22 + tmp24
    tmp27 = tmp26 * tmp12
    tmp28 = tmp25 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp33 = tmp32 * tmp1
    tmp35 = tmp34 * tmp4
    tmp36 = tmp33 + tmp35
    tmp38 = tmp37 * tmp8
    tmp39 = tmp36 + tmp38
    tmp41 = tmp40 * tmp12
    tmp42 = tmp39 + tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.sum(tmp43, 1)[:, None]
    tmp47 = tmp46 * tmp1
    tmp49 = tmp48 * tmp4
    tmp50 = tmp47 + tmp49
    tmp52 = tmp51 * tmp8
    tmp53 = tmp50 + tmp52
    tmp55 = tmp54 * tmp12
    tmp56 = tmp53 + tmp55
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.sum(tmp57, 1)[:, None]
    tmp61 = tmp0 * tmp60
    tmp63 = tmp3 * tmp62
    tmp64 = tmp61 + tmp63
    tmp66 = tmp7 * tmp65
    tmp67 = tmp64 + tmp66
    tmp69 = tmp11 * tmp68
    tmp70 = tmp67 + tmp69
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
    tmp73 = tl.sum(tmp71, 1)[:, None]
    tmp74 = tmp18 * tmp60
    tmp75 = tmp20 * tmp62
    tmp76 = tmp74 + tmp75
    tmp77 = tmp23 * tmp65
    tmp78 = tmp76 + tmp77
    tmp79 = tmp26 * tmp68
    tmp80 = tmp78 + tmp79
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp83 = tl.sum(tmp81, 1)[:, None]
    tmp84 = tmp32 * tmp60
    tmp85 = tmp34 * tmp62
    tmp86 = tmp84 + tmp85
    tmp87 = tmp37 * tmp65
    tmp88 = tmp86 + tmp87
    tmp89 = tmp40 * tmp68
    tmp90 = tmp88 + tmp89
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK, RBLOCK])
    tmp93 = tl.sum(tmp91, 1)[:, None]
    tmp94 = tmp46 * tmp60
    tmp95 = tmp48 * tmp62
    tmp96 = tmp94 + tmp95
    tmp97 = tmp51 * tmp65
    tmp98 = tmp96 + tmp97
    tmp99 = tmp54 * tmp68
    tmp100 = tmp98 + tmp99
    tmp101 = tl.broadcast_to(tmp100, [XBLOCK, RBLOCK])
    tmp103 = tl.sum(tmp101, 1)[:, None]
    tmp105 = tmp0 * tmp104
    tmp107 = tmp3 * tmp106
    tmp108 = tmp105 + tmp107
    tmp110 = tmp7 * tmp109
    tmp111 = tmp108 + tmp110
    tmp113 = tmp11 * tmp112
    tmp114 = tmp111 + tmp113
    tmp115 = tl.broadcast_to(tmp114, [XBLOCK, RBLOCK])
    tmp117 = tl.sum(tmp115, 1)[:, None]
    tmp118 = tmp18 * tmp104
    tmp119 = tmp20 * tmp106
    tmp120 = tmp118 + tmp119
    tmp121 = tmp23 * tmp109
    tmp122 = tmp120 + tmp121
    tmp123 = tmp26 * tmp112
    tmp124 = tmp122 + tmp123
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp127 = tl.sum(tmp125, 1)[:, None]
    tmp128 = tmp32 * tmp104
    tmp129 = tmp34 * tmp106
    tmp130 = tmp128 + tmp129
    tmp131 = tmp37 * tmp109
    tmp132 = tmp130 + tmp131
    tmp133 = tmp40 * tmp112
    tmp134 = tmp132 + tmp133
    tmp135 = tl.broadcast_to(tmp134, [XBLOCK, RBLOCK])
    tmp137 = tl.sum(tmp135, 1)[:, None]
    tmp138 = tmp46 * tmp104
    tmp139 = tmp48 * tmp106
    tmp140 = tmp138 + tmp139
    tmp141 = tmp51 * tmp109
    tmp142 = tmp140 + tmp141
    tmp143 = tmp54 * tmp112
    tmp144 = tmp142 + tmp143
    tmp145 = tl.broadcast_to(tmp144, [XBLOCK, RBLOCK])
    tmp147 = tl.sum(tmp145, 1)[:, None]
    tmp149 = tmp0 * tmp148
    tmp151 = tmp3 * tmp150
    tmp152 = tmp149 + tmp151
    tmp154 = tmp7 * tmp153
    tmp155 = tmp152 + tmp154
    tmp157 = tmp11 * tmp156
    tmp158 = tmp155 + tmp157
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, RBLOCK])
    tmp161 = tl.sum(tmp159, 1)[:, None]
    tmp162 = tmp18 * tmp148
    tmp163 = tmp20 * tmp150
    tmp164 = tmp162 + tmp163
    tmp165 = tmp23 * tmp153
    tmp166 = tmp164 + tmp165
    tmp167 = tmp26 * tmp156
    tmp168 = tmp166 + tmp167
    tmp169 = tl.broadcast_to(tmp168, [XBLOCK, RBLOCK])
    tmp171 = tl.sum(tmp169, 1)[:, None]
    tmp172 = tmp32 * tmp148
    tmp173 = tmp34 * tmp150
    tmp174 = tmp172 + tmp173
    tmp175 = tmp37 * tmp153
    tmp176 = tmp174 + tmp175
    tmp177 = tmp40 * tmp156
    tmp178 = tmp176 + tmp177
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK, RBLOCK])
    tmp181 = tl.sum(tmp179, 1)[:, None]
    tmp182 = tmp46 * tmp148
    tmp183 = tmp48 * tmp150
    tmp184 = tmp182 + tmp183
    tmp185 = tmp51 * tmp153
    tmp186 = tmp184 + tmp185
    tmp187 = tmp54 * tmp156
    tmp188 = tmp186 + tmp187
    tmp189 = tl.broadcast_to(tmp188, [XBLOCK, RBLOCK])
    tmp191 = tl.sum(tmp189, 1)[:, None]
    tmp192 = 16.0
    tmp193 = tmp17 / tmp192
    tmp194 = 0.0
    tmp195 = tmp194 - tmp193
    tmp196 = tmp31 / tmp192
    tmp197 = tmp195 - tmp196
    tmp198 = tmp45 / tmp192
    tmp199 = tmp197 - tmp198
    tmp200 = tmp59 / tmp192
    tmp201 = tmp199 - tmp200
    tmp202 = tmp73 / tmp192
    tmp203 = tmp201 - tmp202
    tmp204 = tmp83 / tmp192
    tmp205 = tmp203 - tmp204
    tmp206 = tmp93 / tmp192
    tmp207 = tmp205 - tmp206
    tmp208 = tmp103 / tmp192
    tmp209 = tmp207 - tmp208
    tmp210 = tmp117 / tmp192
    tmp211 = tmp209 - tmp210
    tmp212 = tmp127 / tmp192
    tmp213 = tmp211 - tmp212
    tmp214 = tmp137 / tmp192
    tmp215 = tmp213 - tmp214
    tmp216 = tmp147 / tmp192
    tmp217 = tmp215 - tmp216
    tmp218 = tmp161 / tmp192
    tmp219 = tmp217 - tmp218
    tmp220 = tmp171 / tmp192
    tmp221 = tmp219 - tmp220
    tmp222 = tmp181 / tmp192
    tmp223 = tmp221 - tmp222
    tmp224 = tmp191 / tmp192
    tmp225 = tmp223 - tmp224
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp225, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(arg0_1, buf0, 64, grid=grid(64), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lsm], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_1.run(buf0, buf1, 64, grid=grid(64), stream=stream0)
        buf12 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(arg0_1, buf12, 64, grid=grid(64), stream=stream0)
        buf13 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lsm_2], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_1.run(buf12, buf13, 64, grid=grid(64), stream=stream0)
        buf18 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(arg0_1, buf18, 64, grid=grid(64), stream=stream0)
        buf19 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lsm_3], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_1.run(buf18, buf19, 64, grid=grid(64), stream=stream0)
        buf6 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(arg0_1, buf6, 64, grid=grid(64), stream=stream0)
        del arg0_1
        buf7 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lsm_1], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__log_softmax_1.run(buf6, buf7, 64, grid=grid(64), stream=stream0)
        del buf6
        buf2 = empty_strided_cuda((), (), torch.float32)
        buf24 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [mul, loss, mean, total_loss, mul_1, loss_1, mean_1, total_loss_1, mul_2, loss_2, mean_2, total_loss_2, mul_3, loss_3, mean_3, total_loss_3, mul_4, loss_4, mean_4, total_loss_4, mul_5, loss_5, mean_5, total_loss_5, mul_6, loss_6, mean_6, total_loss_6, mul_7, loss_7, mean_7, total_loss_7, mul_8, loss_8, mean_8, total_loss_8, mul_9, loss_9, mean_9, total_loss_9, mul_10, loss_10, mean_10, total_loss_10, mul_11, loss_11, mean_11, total_loss_11, mul_12, loss_12, mean_12, total_loss_12, mul_13, loss_13, mean_13, total_loss_13, mul_14, loss_14, mean_14, total_loss_14, mul_15, loss_15, mean_15, total_loss_15], Original ATen: [aten.mul, aten.sum, aten.mean, aten.rsub, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_rsub_sub_sum_5.run(buf24, arg1_1, buf1, buf7, buf13, buf19, 1, 16, grid=grid(1), stream=stream0)
        del arg1_1
        del buf1
        del buf13
        del buf19
        del buf7
    return (buf24, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
