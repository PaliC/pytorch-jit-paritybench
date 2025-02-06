# AOT ID: ['40_forward']
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


# kernel path: inductor_cache/jx/cjxjs7el6r2q5otkjcv2xyjnmrg6hcolfctx26nbe7w6juqomfty.py
# Topologically Sorted Source Nodes: [cores], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   cores => div_1
# Graph fragment:
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_2, %expand_1), kwargs = {})
triton_poi_fused_div_0 = async_compile.triton('triton_poi_fused_div_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28
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
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bk/cbkmzt2opdoscuvfhby5kap6cgyffxhdol3twqds32wkf5k6aqst.py
# Topologically Sorted Source Nodes: [items], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   items => div_2
# Graph fragment:
#   %div_2 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_3, %expand_2), kwargs = {})
triton_poi_fused_div_1 = async_compile.triton('triton_poi_fused_div_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ug/cugatmgyih3ue7sbrwgrzzz5rxbfnbt5xh2mrhg5rcnntsfie35h.py
# Topologically Sorted Source Nodes: [cates_logits, cates_sample, cates_mode], Original ATen: [aten.div, aten.log, aten.neg, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   cates_logits => div_3
#   cates_mode => exp_1, sum_5
#   cates_sample => add, exp, log, neg, sum_4
# Graph fragment:
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 0.1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%exponential,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%log,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %neg), kwargs = {})
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [-1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 1), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 0.1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
triton_poi_fused__softmax_add_div_log_neg_2 = async_compile.triton('triton_poi_fused__softmax_add_div_log_neg_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_div_log_neg_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_div_log_neg_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (7*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr1 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr0 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr1 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr1 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr0 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr1 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tmp4 = tl_math.log(tmp3)
    tmp5 = -tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp9 * tmp1
    tmp12 = tl_math.log(tmp11)
    tmp13 = -tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tmp14 * tmp7
    tmp16 = triton_helpers.maximum(tmp8, tmp15)
    tmp18 = tmp17 * tmp1
    tmp20 = tl_math.log(tmp19)
    tmp21 = -tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp22 * tmp7
    tmp24 = triton_helpers.maximum(tmp16, tmp23)
    tmp26 = tmp25 * tmp1
    tmp28 = tl_math.log(tmp27)
    tmp29 = -tmp28
    tmp30 = tmp26 + tmp29
    tmp31 = tmp30 * tmp7
    tmp32 = triton_helpers.maximum(tmp24, tmp31)
    tmp34 = tmp33 * tmp1
    tmp36 = tl_math.log(tmp35)
    tmp37 = -tmp36
    tmp38 = tmp34 + tmp37
    tmp39 = tmp38 * tmp7
    tmp40 = triton_helpers.maximum(tmp32, tmp39)
    tmp42 = tmp41 * tmp1
    tmp44 = tl_math.log(tmp43)
    tmp45 = -tmp44
    tmp46 = tmp42 + tmp45
    tmp47 = tmp46 * tmp7
    tmp48 = triton_helpers.maximum(tmp40, tmp47)
    tmp50 = tmp49 * tmp1
    tmp52 = tl_math.log(tmp51)
    tmp53 = -tmp52
    tmp54 = tmp50 + tmp53
    tmp55 = tmp54 * tmp7
    tmp56 = triton_helpers.maximum(tmp48, tmp55)
    tmp57 = tmp8 - tmp56
    tmp58 = tmp57 * tmp7
    tmp59 = tl_math.exp(tmp58)
    tmp60 = tmp15 - tmp56
    tmp61 = tmp60 * tmp7
    tmp62 = tl_math.exp(tmp61)
    tmp63 = tmp59 + tmp62
    tmp64 = tmp23 - tmp56
    tmp65 = tmp64 * tmp7
    tmp66 = tl_math.exp(tmp65)
    tmp67 = tmp63 + tmp66
    tmp68 = tmp31 - tmp56
    tmp69 = tmp68 * tmp7
    tmp70 = tl_math.exp(tmp69)
    tmp71 = tmp67 + tmp70
    tmp72 = tmp39 - tmp56
    tmp73 = tmp72 * tmp7
    tmp74 = tl_math.exp(tmp73)
    tmp75 = tmp71 + tmp74
    tmp76 = tmp47 - tmp56
    tmp77 = tmp76 * tmp7
    tmp78 = tl_math.exp(tmp77)
    tmp79 = tmp75 + tmp78
    tmp80 = tmp55 - tmp56
    tmp81 = tmp80 * tmp7
    tmp82 = tl_math.exp(tmp81)
    tmp83 = tmp79 + tmp82
    tmp84 = tmp0 * tmp7
    tmp85 = tmp9 * tmp7
    tmp86 = triton_helpers.maximum(tmp84, tmp85)
    tmp87 = tmp17 * tmp7
    tmp88 = triton_helpers.maximum(tmp86, tmp87)
    tmp89 = tmp25 * tmp7
    tmp90 = triton_helpers.maximum(tmp88, tmp89)
    tmp91 = tmp33 * tmp7
    tmp92 = triton_helpers.maximum(tmp90, tmp91)
    tmp93 = tmp41 * tmp7
    tmp94 = triton_helpers.maximum(tmp92, tmp93)
    tmp95 = tmp49 * tmp7
    tmp96 = triton_helpers.maximum(tmp94, tmp95)
    tmp97 = tmp84 - tmp96
    tmp98 = tmp97 * tmp1
    tmp99 = tl_math.exp(tmp98)
    tmp100 = tmp85 - tmp96
    tmp101 = tmp100 * tmp1
    tmp102 = tl_math.exp(tmp101)
    tmp103 = tmp99 + tmp102
    tmp104 = tmp87 - tmp96
    tmp105 = tmp104 * tmp1
    tmp106 = tl_math.exp(tmp105)
    tmp107 = tmp103 + tmp106
    tmp108 = tmp89 - tmp96
    tmp109 = tmp108 * tmp1
    tmp110 = tl_math.exp(tmp109)
    tmp111 = tmp107 + tmp110
    tmp112 = tmp91 - tmp96
    tmp113 = tmp112 * tmp1
    tmp114 = tl_math.exp(tmp113)
    tmp115 = tmp111 + tmp114
    tmp116 = tmp93 - tmp96
    tmp117 = tmp116 * tmp1
    tmp118 = tl_math.exp(tmp117)
    tmp119 = tmp115 + tmp118
    tmp120 = tmp95 - tmp96
    tmp121 = tmp120 * tmp1
    tmp122 = tl_math.exp(tmp121)
    tmp123 = tmp119 + tmp122
    tl.store(out_ptr0 + (x0), tmp56, xmask)
    tl.store(out_ptr1 + (x0), tmp83, xmask)
    tl.store(out_ptr2 + (x0), tmp96, xmask)
    tl.store(out_ptr3 + (x0), tmp123, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/as/cas5ulgz7ptj3jcyylfsiixlxd4435rvkxh6koieaokvtf6zenyt.py
# Topologically Sorted Source Nodes: [cates_logits, cates_sample, cates_mode], Original ATen: [aten.div, aten.log, aten.neg, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   cates_logits => div_3
#   cates_mode => div_6, exp_1
#   cates_sample => add, div_5, exp, log, neg
# Graph fragment:
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, 0.1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%exponential,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%log,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%div_3, %neg), kwargs = {})
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 1), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 1), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_4), kwargs = {})
#   %mul_tensor : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 1), kwargs = {})
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor, %amax_default), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor, 0.1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor,), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_1, %sum_5), kwargs = {})
triton_poi_fused__softmax_add_div_log_neg_3 = async_compile.triton('triton_poi_fused__softmax_add_div_log_neg_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_div_log_neg_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_div_log_neg_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 7
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp9 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tmp4 = tl_math.log(tmp3)
    tmp5 = -tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 1.0
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp11 = tmp10 * tmp7
    tmp12 = tl_math.exp(tmp11)
    tmp14 = tmp12 / tmp13
    tmp15 = tmp0 * tmp7
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17 * tmp1
    tmp19 = tl_math.exp(tmp18)
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (x2), tmp14, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qu/cqucnuktsrrz7wdlcocvxg6joij2aispsivmhg25braz65legoos.py
# Topologically Sorted Source Nodes: [x, x_k, x_k_1, x_k_2, x_k_3, x_k_4, x_k_5, x_k_6], Original ATen: [aten.div, aten.mul]
# Source node to ATen node mapping:
#   x => div
#   x_k => mul_2
#   x_k_1 => mul_4
#   x_k_2 => mul_6
#   x_k_3 => mul_8
#   x_k_4 => mul_10
#   x_k_5 => mul_12
#   x_k_6 => mul_14
# Graph fragment:
#   %div : [num_users=7] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %expand), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_1), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_2), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_3), kwargs = {})
#   %mul_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_4), kwargs = {})
#   %mul_12 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_5), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view_6), kwargs = {})
triton_poi_fused_div_mul_4 = async_compile.triton('triton_poi_fused_div_mul_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 19, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mul_4(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (7*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (7*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr1 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr2 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr2 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr1 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr2 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr1 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr2 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tmp17 = 0.0
    tmp18 = tmp16 * tmp17
    tmp20 = 1.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp23 = tmp15 * tmp22
    tmp25 = tmp24 * tmp17
    tmp27 = tmp26 * tmp20
    tmp28 = tmp25 + tmp27
    tmp29 = tmp15 * tmp28
    tmp31 = tmp30 * tmp17
    tmp33 = tmp32 * tmp20
    tmp34 = tmp31 + tmp33
    tmp35 = tmp15 * tmp34
    tmp37 = tmp36 * tmp17
    tmp39 = tmp38 * tmp20
    tmp40 = tmp37 + tmp39
    tmp41 = tmp15 * tmp40
    tmp43 = tmp42 * tmp17
    tmp45 = tmp44 * tmp20
    tmp46 = tmp43 + tmp45
    tmp47 = tmp15 * tmp46
    tmp49 = tmp48 * tmp17
    tmp51 = tmp50 * tmp20
    tmp52 = tmp49 + tmp51
    tmp53 = tmp15 * tmp52
    tmp55 = tmp54 * tmp17
    tmp57 = tmp56 * tmp20
    tmp58 = tmp55 + tmp57
    tmp59 = tmp15 * tmp58
    tl.store(out_ptr1 + (x2), tmp23, xmask)
    tl.store(out_ptr2 + (x2), tmp29, xmask)
    tl.store(out_ptr3 + (x2), tmp35, xmask)
    tl.store(out_ptr4 + (x2), tmp41, xmask)
    tl.store(out_ptr5 + (x2), tmp47, xmask)
    tl.store(out_ptr6 + (x2), tmp53, xmask)
    tl.store(out_ptr7 + (x2), tmp59, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ji/cjin5vnk7oci3j7uzmylvyeodtut5kpwaf6mrbtlmksv7sa5lovp.py
# Topologically Sorted Source Nodes: [z_k], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   z_k => div_7
# Graph fragment:
#   %div_7 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%slice_3, %expand_3), kwargs = {})
triton_poi_fused_div_5 = async_compile.triton('triton_poi_fused_div_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (8*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 8*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 8*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 8*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oe/coeeinna74zkaakmhqfuvfuyid7c7fb2zdhcda2mm6pict23yc2v.py
# Topologically Sorted Source Nodes: [logits_k, probs_k, probs_k_1, logits_k_1, probs_k_2, probs_k_3, probs, logits_k_2, probs_k_4, probs_k_5, probs_1, logits_k_3, probs_k_6, probs_k_7, probs_2, logits_k_4, probs_k_8, probs_k_9, probs_3, logits_k_5, probs_k_10, probs_k_11, probs_4, logits_k_6, probs_k_12, probs_k_13, probs_5, logits], Original ATen: [aten.div, aten.exp, aten.mul, aten.add, aten.log]
# Source node to ATen node mapping:
#   logits => log_1
#   logits_k => div_8
#   logits_k_1 => div_10
#   logits_k_2 => div_12
#   logits_k_3 => div_14
#   logits_k_4 => div_16
#   logits_k_5 => div_18
#   logits_k_6 => div_20
#   probs => add_2
#   probs_1 => add_3
#   probs_2 => add_4
#   probs_3 => add_5
#   probs_4 => add_6
#   probs_5 => add_7
#   probs_k => exp_2
#   probs_k_1 => mul_3
#   probs_k_10 => exp_7
#   probs_k_11 => mul_13
#   probs_k_12 => exp_8
#   probs_k_13 => mul_15
#   probs_k_2 => exp_3
#   probs_k_3 => mul_5
#   probs_k_4 => exp_4
#   probs_k_5 => mul_7
#   probs_k_6 => exp_5
#   probs_k_7 => mul_9
#   probs_k_8 => exp_6
#   probs_k_9 => mul_11
# Graph fragment:
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_1, 0.1), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_8,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_2, %view), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_2, 0.1), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_10,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_3, %view_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %mul_5), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_3, 0.1), kwargs = {})
#   %exp_4 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_12,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_4, %view_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_7), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_4, 0.1), kwargs = {})
#   %exp_5 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_14,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_5, %view_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %mul_9), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_5, 0.1), kwargs = {})
#   %exp_6 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_16,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_6, %view_4), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %mul_11), kwargs = {})
#   %div_18 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_6, 0.1), kwargs = {})
#   %exp_7 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_18,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_7, %view_5), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_13), kwargs = {})
#   %div_20 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm_7, 0.1), kwargs = {})
#   %exp_8 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%div_20,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_8, %view_6), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %mul_15), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_7,), kwargs = {})
triton_poi_fused_add_div_exp_log_mul_6 = async_compile.triton('triton_poi_fused_add_div_exp_log_mul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_exp_log_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_exp_log_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (7*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (7*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x2), xmask)
    tmp15 = tl.load(in_ptr1 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (1 + 7*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr4 + (x2), xmask)
    tmp25 = tl.load(in_ptr1 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (2 + 7*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr5 + (x2), xmask)
    tmp35 = tl.load(in_ptr1 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr2 + (3 + 7*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr6 + (x2), xmask)
    tmp45 = tl.load(in_ptr1 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr2 + (4 + 7*x0), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr7 + (x2), xmask)
    tmp55 = tl.load(in_ptr1 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr2 + (5 + 7*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr8 + (x2), xmask)
    tmp65 = tl.load(in_ptr1 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr2 + (6 + 7*x0), xmask, eviction_policy='evict_last')
    tmp1 = 10.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl_math.exp(tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 * tmp5
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp3 * tmp10
    tmp13 = tmp12 * tmp1
    tmp14 = tl_math.exp(tmp13)
    tmp16 = tmp15 * tmp5
    tmp18 = tmp17 * tmp8
    tmp19 = tmp16 + tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = tmp11 + tmp20
    tmp23 = tmp22 * tmp1
    tmp24 = tl_math.exp(tmp23)
    tmp26 = tmp25 * tmp5
    tmp28 = tmp27 * tmp8
    tmp29 = tmp26 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = tmp21 + tmp30
    tmp33 = tmp32 * tmp1
    tmp34 = tl_math.exp(tmp33)
    tmp36 = tmp35 * tmp5
    tmp38 = tmp37 * tmp8
    tmp39 = tmp36 + tmp38
    tmp40 = tmp34 * tmp39
    tmp41 = tmp31 + tmp40
    tmp43 = tmp42 * tmp1
    tmp44 = tl_math.exp(tmp43)
    tmp46 = tmp45 * tmp5
    tmp48 = tmp47 * tmp8
    tmp49 = tmp46 + tmp48
    tmp50 = tmp44 * tmp49
    tmp51 = tmp41 + tmp50
    tmp53 = tmp52 * tmp1
    tmp54 = tl_math.exp(tmp53)
    tmp56 = tmp55 * tmp5
    tmp58 = tmp57 * tmp8
    tmp59 = tmp56 + tmp58
    tmp60 = tmp54 * tmp59
    tmp61 = tmp51 + tmp60
    tmp63 = tmp62 * tmp1
    tmp64 = tl_math.exp(tmp63)
    tmp66 = tmp65 * tmp5
    tmp68 = tmp67 * tmp8
    tmp69 = tmp66 + tmp68
    tmp70 = tmp64 * tmp69
    tmp71 = tmp61 + tmp70
    tmp72 = tl_math.log(tmp71)
    tl.store(in_out_ptr0 + (x2), tmp71, xmask)
    tl.store(out_ptr0 + (x2), tmp72, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (7, 4), (4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (8, 4), (4, 1))
    assert_size_stride(primals_5, (8, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((7, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cores], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_0.run(primals_2, buf1, 28, grid=grid(28), stream=stream0)
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [items], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_1.run(primals_3, buf2, 16, grid=grid(16), stream=stream0)
        buf3 = empty_strided_cuda((4, 7), (7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(buf2, reinterpret_tensor(buf1, (4, 7), (1, 4), 0), out=buf3)
        buf4 = empty_strided_cuda((4, 7), (7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cates_sample], Original ATen: [aten.exponential]
        buf5 = torch.ops.aten.exponential.default(buf4)
        buf6 = buf5
        del buf5
        buf7 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf8 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf10 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf11 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [cates_logits, cates_sample, cates_mode], Original ATen: [aten.div, aten.log, aten.neg, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_div_log_neg_2.run(buf3, buf6, buf7, buf8, buf10, buf11, 4, grid=grid(4), stream=stream0)
        buf9 = buf6; del buf6  # reuse
        buf12 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [cates_logits, cates_sample, cates_mode], Original ATen: [aten.div, aten.log, aten.neg, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_div_log_neg_3.run(buf9, buf3, buf7, buf8, buf10, buf11, buf12, 28, grid=grid(28), stream=stream0)
        del buf10
        del buf11
        del buf3
        del buf7
        del buf8
        buf13 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf21 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf26 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf30 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf39 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_k, x_k_1, x_k_2, x_k_3, x_k_4, x_k_5, x_k_6], Original ATen: [aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_mul_4.run(primals_1, buf9, buf12, buf13, buf17, buf21, buf26, buf30, buf34, buf39, 16, grid=grid(16), stream=stream0)
        buf14 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf13, reinterpret_tensor(primals_4, (4, 8), (1, 4), 0), alpha=1, beta=1, out=buf14)
        buf15 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z_k], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf14, buf15, 16, grid=grid(16), stream=stream0)
        buf16 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf15, reinterpret_tensor(buf2, (4, 4), (1, 4), 0), out=buf16)
        buf18 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf17, reinterpret_tensor(primals_4, (4, 8), (1, 4), 0), alpha=1, beta=1, out=buf18)
        buf19 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z_k_1], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf18, buf19, 16, grid=grid(16), stream=stream0)
        buf20 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf19, reinterpret_tensor(buf2, (4, 4), (1, 4), 0), out=buf20)
        buf22 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf21, reinterpret_tensor(primals_4, (4, 8), (1, 4), 0), alpha=1, beta=1, out=buf22)
        buf23 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z_k_2], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf22, buf23, 16, grid=grid(16), stream=stream0)
        buf24 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf23, reinterpret_tensor(buf2, (4, 4), (1, 4), 0), out=buf24)
        buf27 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf26, reinterpret_tensor(primals_4, (4, 8), (1, 4), 0), alpha=1, beta=1, out=buf27)
        buf28 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z_k_3], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf27, buf28, 16, grid=grid(16), stream=stream0)
        buf29 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf28, reinterpret_tensor(buf2, (4, 4), (1, 4), 0), out=buf29)
        buf31 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf30, reinterpret_tensor(primals_4, (4, 8), (1, 4), 0), alpha=1, beta=1, out=buf31)
        buf32 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z_k_4], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf31, buf32, 16, grid=grid(16), stream=stream0)
        buf33 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf32, reinterpret_tensor(buf2, (4, 4), (1, 4), 0), out=buf33)
        buf35 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf34, reinterpret_tensor(primals_4, (4, 8), (1, 4), 0), alpha=1, beta=1, out=buf35)
        buf36 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z_k_5], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf35, buf36, 16, grid=grid(16), stream=stream0)
        buf37 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf36, reinterpret_tensor(buf2, (4, 4), (1, 4), 0), out=buf37)
        buf40 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf39, reinterpret_tensor(primals_4, (4, 8), (1, 4), 0), alpha=1, beta=1, out=buf40)
        del primals_5
        buf41 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z_k_6], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf40, buf41, 16, grid=grid(16), stream=stream0)
        buf42 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf41, reinterpret_tensor(buf2, (4, 4), (1, 4), 0), out=buf42)
        buf25 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        buf38 = buf25; del buf25  # reuse
        buf43 = buf38; del buf38  # reuse
        buf44 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits_k, probs_k, probs_k_1, logits_k_1, probs_k_2, probs_k_3, probs, logits_k_2, probs_k_4, probs_k_5, probs_1, logits_k_3, probs_k_6, probs_k_7, probs_2, logits_k_4, probs_k_8, probs_k_9, probs_3, logits_k_5, probs_k_10, probs_k_11, probs_4, logits_k_6, probs_k_12, probs_k_13, probs_5, logits], Original ATen: [aten.div, aten.exp, aten.mul, aten.add, aten.log]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_log_mul_6.run(buf43, buf16, buf9, buf12, buf20, buf24, buf29, buf33, buf37, buf42, buf44, 16, grid=grid(16), stream=stream0)
    return (buf44, reinterpret_tensor(buf14, (4, 4), (8, 1), 0), reinterpret_tensor(buf18, (4, 4), (8, 1), 0), reinterpret_tensor(buf22, (4, 4), (8, 1), 0), reinterpret_tensor(buf27, (4, 4), (8, 1), 0), reinterpret_tensor(buf31, (4, 4), (8, 1), 0), reinterpret_tensor(buf35, (4, 4), (8, 1), 0), reinterpret_tensor(buf40, (4, 4), (8, 1), 0), reinterpret_tensor(buf14, (4, 4), (8, 1), 4), reinterpret_tensor(buf18, (4, 4), (8, 1), 4), reinterpret_tensor(buf22, (4, 4), (8, 1), 4), reinterpret_tensor(buf27, (4, 4), (8, 1), 4), reinterpret_tensor(buf31, (4, 4), (8, 1), 4), reinterpret_tensor(buf35, (4, 4), (8, 1), 4), reinterpret_tensor(buf40, (4, 4), (8, 1), 4), buf14, buf18, buf22, buf27, buf31, buf35, buf40, primals_1, primals_2, primals_3, buf1, buf2, buf9, buf12, buf13, reinterpret_tensor(buf14, (4, 4), (8, 1), 0), buf15, buf16, buf17, reinterpret_tensor(buf18, (4, 4), (8, 1), 0), buf19, buf20, buf21, reinterpret_tensor(buf22, (4, 4), (8, 1), 0), buf23, buf24, buf26, reinterpret_tensor(buf27, (4, 4), (8, 1), 0), buf28, buf29, buf30, reinterpret_tensor(buf31, (4, 4), (8, 1), 0), buf32, buf33, buf34, reinterpret_tensor(buf35, (4, 4), (8, 1), 0), buf36, buf37, buf39, reinterpret_tensor(buf40, (4, 4), (8, 1), 0), buf41, buf42, buf43, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((7, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
