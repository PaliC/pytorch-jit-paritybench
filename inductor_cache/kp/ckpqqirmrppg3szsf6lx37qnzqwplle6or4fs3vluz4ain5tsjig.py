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


# kernel path: inductor_cache/ks/cks33qhizchbf3ja2zohtnviprb5xwql672h45hvjutivwpm5rd4.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [1]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_poi_fused_native_layer_norm_0 = async_compile.triton('triton_poi_fused_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n3/cn3sixhqcwo3n77lh2cad7nysgt7immngfvzlk3th3575uvlwina.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [1]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_4), kwargs = {})
triton_poi_fused_native_layer_norm_1 = async_compile.triton('triton_poi_fused_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s6/cs65xbduqwdzk4mj7svvsbxtormzc6ud4y6bl4qp7h6xlkqb2thc.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   multi_head_attention_forward => mul_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%view_9, 1.0), kwargs = {})
triton_poi_fused_mul_2 = async_compile.triton('triton_poi_fused_mul_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqqlcxdibkttng4op7hssldi627trxj4xywfpvekj75ycrwtrn2.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   multi_head_attention_forward => mul_3
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_6, 1.0), kwargs = {})
triton_poi_fused_mul_3 = async_compile.triton('triton_poi_fused_mul_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ln/clnssapuk34w4dddo7chb3ji43ugh7lam7g5hrbse2z4bplsc75l.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2, amax, any_1, eq, exp, logical_not, sub_1, sum_1
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %unsqueeze_6), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_2, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_2, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
triton_poi_fused__safe_softmax_add_4 = async_compile.triton('triton_poi_fused__safe_softmax_add_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp6 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp11 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp16 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = float("-inf")
    tmp3 = 0.0
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp5 = tmp0 + tmp4
    tmp8 = tl.where(tmp7, tmp2, tmp3)
    tmp9 = tmp6 + tmp8
    tmp10 = triton_helpers.maximum(tmp5, tmp9)
    tmp13 = tl.where(tmp12, tmp2, tmp3)
    tmp14 = tmp11 + tmp13
    tmp15 = triton_helpers.maximum(tmp10, tmp14)
    tmp18 = tl.where(tmp17, tmp2, tmp3)
    tmp19 = tmp16 + tmp18
    tmp20 = triton_helpers.maximum(tmp15, tmp19)
    tmp21 = tmp5 - tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp9 - tmp20
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tmp22 + tmp24
    tmp26 = tmp14 - tmp20
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tmp25 + tmp27
    tmp29 = tmp19 - tmp20
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp28 + tmp30
    tmp32 = tmp5 == tmp2
    tmp33 = tmp32 == 0
    tmp34 = tmp33.to(tl.int64)
    tmp35 = (tmp34 != 0)
    tmp36 = tmp9 == tmp2
    tmp37 = tmp36 == 0
    tmp38 = tmp37.to(tl.int64)
    tmp39 = (tmp38 != 0)
    tmp40 = tmp35 | tmp39
    tmp41 = tmp14 == tmp2
    tmp42 = tmp41 == 0
    tmp43 = tmp42.to(tl.int64)
    tmp44 = (tmp43 != 0)
    tmp45 = tmp40 | tmp44
    tmp46 = tmp19 == tmp2
    tmp47 = tmp46 == 0
    tmp48 = tmp47.to(tl.int64)
    tmp49 = (tmp48 != 0)
    tmp50 = tmp45 | tmp49
    tl.store(out_ptr0 + (x2), tmp20, xmask)
    tl.store(out_ptr1 + (x2), tmp31, xmask)
    tl.store(out_ptr2 + (x2), tmp50, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rr/crrz5426k4bnzq326dgvaq5o5zvoo7bpuqghg5i6bd2ycmijmuua.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   multi_head_attention_forward => add_2, amax, div, exp, full_default_2, logical_not_1, sub_1, where_1
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_14, %unsqueeze_6), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_2, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_2 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 4, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default_2, %div), kwargs = {})
triton_poi_fused__safe_softmax_add_5 = async_compile.triton('triton_poi_fused__safe_softmax_add_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_add_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 4
    x4 = xindex
    x5 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp3 = tl.load(in_ptr1 + (x5), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp8 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 == 0
    tmp4 = float("-inf")
    tmp5 = 0.0
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp2 + tmp6
    tmp9 = tmp7 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp12 = tmp10 / tmp11
    tmp13 = tl.where(tmp1, tmp5, tmp12)
    tl.store(in_out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bq/cbqr243sqqmlcio6smwc6lanoj2ggsbgtolesb2pgklr2bkqdxo6.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ar/car3ben25wmilkyihixtl6xw4w3bt6wxqmsysadjv3nkva7yhuev.py
# Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => var_mean_1
#   x => add_3
# Graph fragment:
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %squeeze), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [1]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_add_native_layer_norm_7 = async_compile.triton('triton_poi_fused_add_native_layer_norm_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_7(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp2 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp5 - tmp16
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp9 - tmp16
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp13 - tmp16
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 / tmp15
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cx/ccxgdpcnvtbtgmjdhfxvfk3iqx5hd62slbg27w64bwgqlqrsz6vr.py
# Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_4, add_5, mul_4, mul_5, rsqrt_1, sub_2
#   x => add_3
# Graph fragment:
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %squeeze), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %primals_9), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %primals_10), kwargs = {})
triton_poi_fused_add_native_layer_norm_8 = async_compile.triton('triton_poi_fused_add_native_layer_norm_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nc/cnce7wgvt2uvpss445rytvznlfwitlugog2jfl4q4bwwysxedxe4.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_6, erf, mul_6, mul_7, mul_8
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_4, 0.5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_4, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_7,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %add_6), kwargs = {})
triton_poi_fused_gelu_9 = async_compile.triton('triton_poi_fused_gelu_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bi/cbikyrdyaakq73is2khvx63mfk7toenq62vpc66qn2jnoktg4lta.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.add, aten.addmm]
# Source node to ATen node mapping:
#   x => add_3
#   x_1 => add_tensor
#   x_2 => add_7
# Graph fragment:
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %squeeze), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_14), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %add_tensor), kwargs = {})
triton_poi_fused_add_addmm_10 = async_compile.triton('triton_poi_fused_add_addmm_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tl.store(in_out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (12, 4), (4, 1))
    assert_size_stride(primals_6, (12, ), (1, ))
    assert_size_stride(primals_7, (4, 4), (4, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (16, 4), (4, 1))
    assert_size_stride(primals_12, (16, ), (1, ))
    assert_size_stride(primals_13, (4, 16), (16, 1))
    assert_size_stride(primals_14, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf1 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_0.run(primals_1, buf0, buf1, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_1.run(primals_1, buf0, buf1, primals_3, primals_4, buf2, 16, grid=grid(16), stream=stream0)
        del primals_3
        del primals_4
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(buf2, reinterpret_tensor(primals_5, (4, 4), (1, 4), 0), out=buf3)
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(buf2, reinterpret_tensor(primals_5, (4, 4), (1, 4), 16), out=buf4)
        buf5 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(primals_6, (4, ), (1, ), 8), buf2, reinterpret_tensor(primals_5, (4, 4), (1, 4), 32), alpha=1, beta=1, out=buf5)
        buf6 = reinterpret_tensor(buf3, (1, 4, 4, 1), (16, 1, 4, 16), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_2.run(buf6, primals_6, 16, grid=grid(16), stream=stream0)
        buf7 = reinterpret_tensor(buf4, (1, 4, 1, 4), (16, 1, 16, 4), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_3.run(buf7, primals_6, 16, grid=grid(16), stream=stream0)
        del primals_6
        buf8 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (4, 4, 1), (1, 4, 0), 0), reinterpret_tensor(buf7, (4, 1, 4), (1, 0, 4), 0), out=buf8)
        buf9 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        buf10 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.float32)
        buf11 = empty_strided_cuda((1, 4, 4, 1), (16, 4, 1, 16), torch.bool)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_add_4.run(buf8, primals_2, buf9, buf10, buf11, 16, grid=grid(16), stream=stream0)
        buf12 = reinterpret_tensor(buf8, (1, 4, 4, 4), (64, 16, 4, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_add_5.run(buf12, buf11, primals_2, buf9, buf10, 64, grid=grid(64), stream=stream0)
        del buf11
        del primals_2
        buf13 = reinterpret_tensor(buf9, (4, 4, 1), (4, 1, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf12, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf5, (4, 4, 1), (1, 4, 0), 0), out=buf13)
        buf14 = reinterpret_tensor(buf10, (4, 1, 4, 1), (4, 1, 1, 4), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf13, buf14, 4, 4, grid=grid(4, 4), stream=stream0)
        buf15 = reinterpret_tensor(buf13, (4, 4), (4, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, reinterpret_tensor(buf14, (4, 4), (4, 1), 0), reinterpret_tensor(primals_7, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf15)
        del primals_8
        buf16 = buf1; del buf1  # reuse
        buf17 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_7.run(primals_1, buf15, buf16, buf17, 4, grid=grid(4), stream=stream0)
        buf18 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_8.run(primals_1, buf15, buf16, buf17, primals_9, primals_10, buf18, 16, grid=grid(16), stream=stream0)
        del buf16
        del buf17
        del primals_10
        buf19 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, buf18, reinterpret_tensor(primals_11, (4, 16), (1, 4), 0), alpha=1, beta=1, out=buf19)
        del primals_12
        buf20 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_9.run(buf19, buf20, 64, grid=grid(64), stream=stream0)
        buf21 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf20, reinterpret_tensor(primals_13, (16, 4), (1, 16), 0), out=buf21)
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.add, aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_10.run(buf22, primals_1, buf15, primals_14, 16, grid=grid(16), stream=stream0)
        del primals_14
    return (buf22, primals_1, primals_9, buf2, buf12, reinterpret_tensor(buf14, (4, 4), (4, 1), 0), buf15, buf18, buf19, buf20, primals_13, primals_11, primals_7, reinterpret_tensor(buf5, (4, 1, 4), (1, 1, 4), 0), reinterpret_tensor(buf6, (4, 1, 4), (1, 4, 4), 0), reinterpret_tensor(buf7, (4, 4, 1), (1, 4, 16), 0), reinterpret_tensor(primals_5, (4, 4), (4, 1), 32), reinterpret_tensor(primals_5, (4, 4), (4, 1), 16), reinterpret_tensor(primals_5, (4, 4), (4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.bool)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((12, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
