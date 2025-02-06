# AOT ID: ['33_forward']
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


# kernel path: inductor_cache/kz/ckzav56m6x4e5ozwrluwzajfeuvjyj7dznixacmxfmyrghr4orz6.py
# Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x => convolution
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [2, 2], [3, 3], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 1024) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/br/cbr67f4fimcgukvib6px57jn5xmt22fdwokdlr3pfvw36l2fbsgi.py
# Topologically Sorted Source Nodes: [x_3, input_1], Original ATen: [aten.max_pool2d_with_indices, aten.mean]
# Source node to ATen node mapping:
#   input_1 => mean
#   x_3 => getitem, getitem_1
# Graph fragment:
#   %getitem : [num_users=4] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%getitem, [-1, -2], True), kwargs = {})
triton_red_fused_max_pool2d_with_indices_mean_1 = async_compile.triton('triton_red_fused_max_pool2d_with_indices_mean_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_max_pool2d_with_indices_mean_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_max_pool2d_with_indices_mean_1(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp78 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex // 16
        r1 = (rindex % 16)
        r3 = rindex
        tmp0 = (-1) + 2*r2
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tmp2 & tmp4
        tmp6 = (-1) + 2*r1
        tmp7 = tmp6 >= tmp1
        tmp8 = tmp6 < tmp3
        tmp9 = tmp7 & tmp8
        tmp10 = tmp5 & tmp9
        tmp11 = tl.load(in_ptr0 + ((-33) + 2*r1 + 64*r2 + 1024*x0), rmask & tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp12 = 2*r1
        tmp13 = tmp12 >= tmp1
        tmp14 = tmp12 < tmp3
        tmp15 = tmp13 & tmp14
        tmp16 = tmp5 & tmp15
        tmp17 = tl.load(in_ptr0 + ((-32) + 2*r1 + 64*r2 + 1024*x0), rmask & tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp18 = triton_helpers.maximum(tmp17, tmp11)
        tmp19 = 1 + 2*r1
        tmp20 = tmp19 >= tmp1
        tmp21 = tmp19 < tmp3
        tmp22 = tmp20 & tmp21
        tmp23 = tmp5 & tmp22
        tmp24 = tl.load(in_ptr0 + ((-31) + 2*r1 + 64*r2 + 1024*x0), rmask & tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp25 = triton_helpers.maximum(tmp24, tmp18)
        tmp26 = 2*r2
        tmp27 = tmp26 >= tmp1
        tmp28 = tmp26 < tmp3
        tmp29 = tmp27 & tmp28
        tmp30 = tmp29 & tmp9
        tmp31 = tl.load(in_ptr0 + ((-1) + 2*r1 + 64*r2 + 1024*x0), rmask & tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp32 = triton_helpers.maximum(tmp31, tmp25)
        tmp33 = tmp29 & tmp15
        tmp34 = tl.load(in_ptr0 + (2*r1 + 64*r2 + 1024*x0), rmask & tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp35 = triton_helpers.maximum(tmp34, tmp32)
        tmp36 = tmp29 & tmp22
        tmp37 = tl.load(in_ptr0 + (1 + 2*r1 + 64*r2 + 1024*x0), rmask & tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp38 = triton_helpers.maximum(tmp37, tmp35)
        tmp39 = 1 + 2*r2
        tmp40 = tmp39 >= tmp1
        tmp41 = tmp39 < tmp3
        tmp42 = tmp40 & tmp41
        tmp43 = tmp42 & tmp9
        tmp44 = tl.load(in_ptr0 + (31 + 2*r1 + 64*r2 + 1024*x0), rmask & tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp45 = triton_helpers.maximum(tmp44, tmp38)
        tmp46 = tmp42 & tmp15
        tmp47 = tl.load(in_ptr0 + (32 + 2*r1 + 64*r2 + 1024*x0), rmask & tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp48 = triton_helpers.maximum(tmp47, tmp45)
        tmp49 = tmp42 & tmp22
        tmp50 = tl.load(in_ptr0 + (33 + 2*r1 + 64*r2 + 1024*x0), rmask & tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp51 = triton_helpers.maximum(tmp50, tmp48)
        tmp52 = tmp17 > tmp11
        tmp53 = tl.full([1, 1], 1, tl.int8)
        tmp54 = tl.full([1, 1], 0, tl.int8)
        tmp55 = tl.where(tmp52, tmp53, tmp54)
        tmp56 = tmp24 > tmp18
        tmp57 = tl.full([1, 1], 2, tl.int8)
        tmp58 = tl.where(tmp56, tmp57, tmp55)
        tmp59 = tmp31 > tmp25
        tmp60 = tl.full([1, 1], 3, tl.int8)
        tmp61 = tl.where(tmp59, tmp60, tmp58)
        tmp62 = tmp34 > tmp32
        tmp63 = tl.full([1, 1], 4, tl.int8)
        tmp64 = tl.where(tmp62, tmp63, tmp61)
        tmp65 = tmp37 > tmp35
        tmp66 = tl.full([1, 1], 5, tl.int8)
        tmp67 = tl.where(tmp65, tmp66, tmp64)
        tmp68 = tmp44 > tmp38
        tmp69 = tl.full([1, 1], 6, tl.int8)
        tmp70 = tl.where(tmp68, tmp69, tmp67)
        tmp71 = tmp47 > tmp45
        tmp72 = tl.full([1, 1], 7, tl.int8)
        tmp73 = tl.where(tmp71, tmp72, tmp70)
        tmp74 = tmp50 > tmp48
        tmp75 = tl.full([1, 1], 8, tl.int8)
        tmp76 = tl.where(tmp74, tmp75, tmp73)
        tmp77 = tl.broadcast_to(tmp51, [XBLOCK, RBLOCK])
        tmp79 = _tmp78 + tmp77
        _tmp78 = tl.where(rmask & xmask, tmp79, _tmp78)
        tl.store(out_ptr0 + (r3 + 256*x0), tmp51, rmask & xmask)
        tl.store(out_ptr1 + (r3 + 256*x0), tmp76, rmask & xmask)
    tmp78 = tl.sum(_tmp78, 1)[:, None]
    tmp80 = 256.0
    tmp81 = tmp78 / tmp80
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp81, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kz/ckzmvwcmgpzrknv2evbzic7ayabx7ywzykhtui62rppqlcdzci3h.py
# Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => convolution_1
#   input_3 => add_3, mul_4, mul_5, sub_1
#   input_4 => relu_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_8, %primals_9, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q6/cq6fybi2wbxqsccmri7exkg2dmtbxut7uydje2j6gzgdvcv3q3en.py
# Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_5 => convolution_2
#   input_6 => add_5, mul_7, mul_8, sub_2
#   input_7 => relu_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mr/cmrfxl6x4ko3spbfnshmzdwpbvpnazet4btk2xut67ntdkx7tmsn.py
# Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
# Source node to ATen node mapping:
#   input_10 => sigmoid
#   input_8 => convolution_3
#   input_9 => add_7, mul_10, mul_11, sub_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_2, %primals_20, %primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.sigmoid(tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x0 + 512*x1), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gx/cgxujiclmwou6q7b66gmos6qkqgderrvj36gceac4qyhc273fu3p.py
# Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_5 => add_9, mul_13, mul_14, sub_4
#   x_6 => relu_3
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_9,), kwargs = {})
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
    x3 = xindex
    x1 = ((xindex // 256) % 128)
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


# kernel path: inductor_cache/vi/cvinlu2vg23rtf5apuuf34b4kstkaukqhoktdsusk6njjykl3z6p.py
# Topologically Sorted Source Nodes: [ss], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   ss => repeat
# Graph fragment:
#   %repeat : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%sigmoid, [1, 4, 1, 1]), kwargs = {})
triton_poi_fused_repeat_6 = async_compile.triton('triton_poi_fused_repeat_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512*x1 + ((x0 % 32))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3s/c3ss3qukvgjvfy2bgxcdl3q5itojufvwe4ztn4a5cuwotlehgeoh.py
# Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   x_10 => mul_18
#   x_8 => add_11, mul_16, mul_17, sub_5
#   x_9 => relu_4
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_4 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
#   %mul_18 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %relu_4), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 256) % 128)
    x2 = xindex // 32768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((x1 % 4)) + 128*x2 + (x1 // 4)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 * tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/46/c465zczoojjleqgoz7ezepr5uadpelar4t7ibpfwfzed5che5jhc.py
# Topologically Sorted Source Nodes: [x_12, x_13, input_12, add, x_14, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   add => add_16
#   input_12 => add_15, mul_23, mul_24, sub_7
#   input_13 => mean_1
#   x_12 => add_13, mul_20, mul_21, sub_6
#   x_13 => relu_5
#   x_14 => relu_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_55), kwargs = {})
#   %relu_5 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_13,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %unsqueeze_63), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %relu_5), kwargs = {})
#   %relu_6 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_6, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_8 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (r2 + 256*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (r2 + 256*x3), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp31 = tmp15 + tmp30
    tmp32 = triton_helpers.maximum(tmp29, tmp31)
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = triton_helpers.promote_to_tensor(tl.sum(tmp33, 0))
    tmp36 = 256.0
    tmp37 = tmp35 / tmp36
    tl.store(in_out_ptr0 + (r2 + 256*x3), tmp32, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/jb/cjb3mpdvwjcglely4qe5kcxmhswvjufo3lxrjyrbsyhoefqnbxdl.py
# Topologically Sorted Source Nodes: [x_23, x_24, add_1, x_25, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   add_1 => add_29
#   input_23 => mean_2
#   x_23 => add_28, mul_42, mul_43, sub_13
#   x_24 => relu_11
#   x_25 => relu_12
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_42, %unsqueeze_109), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_43, %unsqueeze_111), kwargs = {})
#   %relu_11 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_6, %relu_11), kwargs = {})
#   %relu_12 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_29,), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_12, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_9 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = tl.full([RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (r2 + 256*x3), None)
    tmp1 = tl.load(in_ptr1 + (r2 + 256*x3), None)
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp24 = 256.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr0 + (r2 + 256*x3), tmp20, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/4i/c4i6xb2r4kaz3eznoilz3mto63l2vqj4ebxrzidrntbft3y5rxuj.py
# Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_34 => convolution_20
#   input_35 => add_44, mul_64, mul_65, sub_20
#   input_36 => relu_19
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_3, %primals_112, %primals_113, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_165), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_167), kwargs = {})
#   %relu_19 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_44,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/si/csi43eofb7wtndap7pis3nzdhooabwml5guvwd3t53t4samatyjz.py
# Topologically Sorted Source Nodes: [x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_38 => add_50, mul_73, mul_74, sub_23
#   x_39 => relu_21
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_189), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_191), kwargs = {})
#   %relu_21 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_50,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 256) % 256)
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


# kernel path: inductor_cache/mg/cmgbeqjk3ei6qfwzt3vrfuf5zryc3zqeuvkwzlv6gz7jfxqqz5me.py
# Topologically Sorted Source Nodes: [ss_12], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   ss_12 => repeat_3
# Graph fragment:
#   %repeat_3 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%sigmoid_3, [1, 8, 1, 1]), kwargs = {})
triton_poi_fused_repeat_12 = async_compile.triton('triton_poi_fused_repeat_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 256)
    x1 = xindex // 256
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512*x1 + ((x0 % 32))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j3/cj3tb5jvvdq6aealit5k3k4gvgf3a6lkbclcvmkjojeyhacpzked.py
# Topologically Sorted Source Nodes: [x_41, x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   x_41 => add_52, mul_76, mul_77, sub_24
#   x_42 => relu_22
#   x_43 => mul_78
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_76 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_77 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_76, %unsqueeze_197), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_77, %unsqueeze_199), kwargs = {})
#   %relu_22 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_52,), kwargs = {})
#   %mul_78 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_7, %relu_22), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 64) % 256)
    x2 = xindex // 16384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((x1 % 8)) + 256*x2 + (x1 // 8)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 * tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/6h/c6h5jfxdwg6fmketcllgwew6rwamordbnqy4iwwgmpal6uy2tmet.py
# Topologically Sorted Source Nodes: [x_45, x_46, input_44, add_3, x_47, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   add_3 => add_57
#   input_44 => add_56, mul_83, mul_84, sub_26
#   input_45 => mean_4
#   x_45 => add_54, mul_80, mul_81, sub_25
#   x_46 => relu_23
#   x_47 => relu_24
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_81 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_80, %unsqueeze_205), kwargs = {})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_81, %unsqueeze_207), kwargs = {})
#   %relu_23 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_54,), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %unsqueeze_213), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %unsqueeze_215), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %relu_23), kwargs = {})
#   %relu_24 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_57,), kwargs = {})
#   %mean_4 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_24, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_14 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (r2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (r2 + 64*x3), xmask, other=0.0)
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.full([1, 1], 0, tl.int32)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp31 = tmp15 + tmp30
    tmp32 = triton_helpers.maximum(tmp29, tmp31)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = 64.0
    tmp38 = tmp36 / tmp37
    tl.store(in_out_ptr0 + (r2 + 64*x3), tmp32, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp38, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wb/cwb4nofejtxcxu635r44jaxfqay65ajdlyicgzkbtqljbeu5qfwy.py
# Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_49 => add_65, mul_95, mul_96, sub_30
#   x_50 => relu_27
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_95 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_95, %unsqueeze_245), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_96, %unsqueeze_247), kwargs = {})
#   %relu_27 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_65,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 256)
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


# kernel path: inductor_cache/kn/cknxxwt27jgtqosokexoqpviy7aw46ic5kewofojtkef6opodqn6.py
# Topologically Sorted Source Nodes: [x_56, x_57, add_4, x_58, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   add_4 => add_70
#   input_55 => mean_5
#   x_56 => add_69, mul_102, mul_103, sub_32
#   x_57 => relu_29
#   x_58 => relu_30
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %mul_102 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_103 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %unsqueeze_261), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_103, %unsqueeze_263), kwargs = {})
#   %relu_29 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_69,), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_24, %relu_29), kwargs = {})
#   %relu_30 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_70,), kwargs = {})
#   %mean_5 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_30, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_16 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (r2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + 64*x3), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
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
    tmp17 = tl.full([1, 1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = 64.0
    tmp26 = tmp24 / tmp25
    tl.store(out_ptr0 + (r2 + 64*x3), tmp20, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/em/cemi37zwgml244nfnfj4s4w5h2nqm7mawksjzmhzigtblgbspswx.py
# Topologically Sorted Source Nodes: [input_76, input_77, input_78], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_76 => convolution_45
#   input_77 => add_98, mul_143, mul_144, sub_45
#   input_78 => relu_43
# Graph fragment:
#   %convolution_45 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mean_7, %primals_249, %primals_250, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_143, %unsqueeze_365), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, %unsqueeze_367), kwargs = {})
#   %relu_43 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_98,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/er/cerlmbhznde57i7whjgspzlc2kfahic6tovazte7n37asktvjbv6.py
# Topologically Sorted Source Nodes: [x_82, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_82 => add_104, mul_152, mul_153, sub_48
#   x_83 => relu_45
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %unsqueeze_385), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_387), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_152, %unsqueeze_389), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_153, %unsqueeze_391), kwargs = {})
#   %relu_45 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_104,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 512)
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


# kernel path: inductor_cache/7p/c7ptimwulqoziotojtplffwthas2yymmw64g5idqijttdwuuazkp.py
# Topologically Sorted Source Nodes: [ss_28], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   ss_28 => repeat_7
# Graph fragment:
#   %repeat_7 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%sigmoid_7, [1, 16, 1, 1]), kwargs = {})
triton_poi_fused_repeat_19 = async_compile.triton('triton_poi_fused_repeat_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 512)
    x1 = xindex // 512
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512*x1 + ((x0 % 32))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ei/ceiy4gdu6zuloopakmhfszjyjhkkrx63npdm6vyntjaxyvltc6bp.py
# Topologically Sorted Source Nodes: [x_85, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   x_85 => add_106, mul_155, mul_156, sub_49
#   x_86 => relu_46
#   x_87 => mul_157
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_393), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_155, %unsqueeze_397), kwargs = {})
#   %add_106 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_156, %unsqueeze_399), kwargs = {})
#   %relu_46 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_106,), kwargs = {})
#   %mul_157 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_15, %relu_46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 512)
    x2 = xindex // 8192
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((x1 % 16)) + 512*x2 + (x1 // 16)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 * tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/yu/cyu4atiykzi647y2yvxigjg2trl64tllwmkerpksx7wbee3qzbec.py
# Topologically Sorted Source Nodes: [x_89, x_90, input_86, add_7, x_91, input_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   add_7 => add_111
#   input_86 => add_110, mul_162, mul_163, sub_51
#   input_87 => mean_8
#   x_89 => add_108, mul_159, mul_160, sub_50
#   x_90 => relu_47
#   x_91 => relu_48
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_401), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_160 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %unsqueeze_405), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_160, %unsqueeze_407), kwargs = {})
#   %relu_47 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_108,), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_409), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %unsqueeze_413), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %unsqueeze_415), kwargs = {})
#   %add_111 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_110, %relu_47), kwargs = {})
#   %relu_48 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_111,), kwargs = {})
#   %mean_8 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_48, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_21 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_21', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_21(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (r2 + 16*x3), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (r2 + 16*x3), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.full([1, 1], 0, tl.int32)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp31 = tmp15 + tmp30
    tmp32 = triton_helpers.maximum(tmp29, tmp31)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.sum(tmp33, 1)[:, None]
    tmp36 = 16.0
    tmp37 = tmp35 / tmp36
    tl.store(in_out_ptr0 + (r2 + 16*x3), tmp32, None)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/xl/cxlpjy6sbvbg6jitlnssafvv2xk6z2663n4bvhya3df3snw3wv3f.py
# Topologically Sorted Source Nodes: [x_93, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_93 => add_119, mul_174, mul_175, sub_55
#   x_94 => relu_51
# Graph fragment:
#   %sub_55 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_55, %unsqueeze_441), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_55, %unsqueeze_443), kwargs = {})
#   %mul_175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_174, %unsqueeze_445), kwargs = {})
#   %add_119 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_175, %unsqueeze_447), kwargs = {})
#   %relu_51 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_119,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/4d/c4dkxty2uvmx5ehyf2w4lepd7hytdjrzcf3ztzktvwnmkzzoulhf.py
# Topologically Sorted Source Nodes: [x_100, x_101, add_8, x_102, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
# Source node to ATen node mapping:
#   add_8 => add_124
#   input_97 => mean_9
#   x_100 => add_123, mul_181, mul_182, sub_57
#   x_101 => relu_53
#   x_102 => relu_54
# Graph fragment:
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_457), kwargs = {})
#   %mul_181 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_182 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_181, %unsqueeze_461), kwargs = {})
#   %add_123 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_182, %unsqueeze_463), kwargs = {})
#   %relu_53 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_123,), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_48, %relu_53), kwargs = {})
#   %relu_54 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_124,), kwargs = {})
#   %mean_9 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_54, [-1, -2], True), kwargs = {})
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23 = async_compile.triton('triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (r2 + 16*x3), None)
    tmp1 = tl.load(in_ptr1 + (r2 + 16*x3), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1, 1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None]
    tmp24 = 16.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr0 + (r2 + 16*x3), tmp20, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp25, None)
''', device_str='cuda')


# kernel path: inductor_cache/os/cos2k3hhacio5mk6i5bcgcbp7aghaajrr6xm64ostb5q2kswbuzl.py
# Topologically Sorted Source Nodes: [x_148, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_148 => add_184, mul_269, mul_270, sub_85
#   x_149 => relu_81
# Graph fragment:
#   %sub_85 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_85, %unsqueeze_681), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_85, %unsqueeze_683), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %unsqueeze_685), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_270, %unsqueeze_687), kwargs = {})
#   %relu_81 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_184,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 16) % 1024)
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


# kernel path: inductor_cache/ad/cadfkoneyionied2tjsiilmayzafwzrfx3cfrvv4q7opkukvmrr6.py
# Topologically Sorted Source Nodes: [ss_52], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   ss_52 => repeat_13
# Graph fragment:
#   %repeat_13 : [num_users=2] = call_function[target=torch.ops.aten.repeat.default](args = (%sigmoid_13, [1, 32, 1, 1]), kwargs = {})
triton_poi_fused_repeat_25 = async_compile.triton('triton_poi_fused_repeat_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (512*x1 + ((x0 % 32))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/mg/cmgl4opmwm2rjqtxrbnbb4md5uecdvamneb5euxvqf6mu3f4vnqy.py
# Topologically Sorted Source Nodes: [x_151, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
# Source node to ATen node mapping:
#   x_151 => add_186, mul_272, mul_273, sub_86
#   x_152 => relu_82
#   x_153 => mul_274
# Graph fragment:
#   %sub_86 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_86, %unsqueeze_689), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_86, %unsqueeze_691), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %unsqueeze_693), kwargs = {})
#   %add_186 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_273, %unsqueeze_695), kwargs = {})
#   %relu_82 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_186,), kwargs = {})
#   %mul_274 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_27, %relu_82), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_26', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 1024)
    x2 = xindex // 4096
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (32*((x1 % 32)) + 1024*x2 + (x1 // 32)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 * tmp18
    tl.store(out_ptr0 + (x3), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/hj/chjimgchxgv35tsabzted3no44ma6au2wdk6ug6acxtz4j47hmq7.py
# Topologically Sorted Source Nodes: [x_155, x_156, input_148, add_13, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_13 => add_191
#   input_148 => add_190, mul_279, mul_280, sub_88
#   x_155 => add_188, mul_276, mul_277, sub_87
#   x_156 => relu_83
#   x_157 => relu_84
# Graph fragment:
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_697), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_277 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_276, %unsqueeze_701), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_277, %unsqueeze_703), kwargs = {})
#   %relu_83 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_188,), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_705), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_279, %unsqueeze_709), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_280, %unsqueeze_711), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_190, %relu_83), kwargs = {})
#   %relu_84 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_191,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 2048)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x3), None)
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = triton_helpers.maximum(tmp29, tmp28)
    tmp31 = tmp15 + tmp30
    tmp32 = triton_helpers.maximum(tmp29, tmp31)
    tl.store(in_out_ptr0 + (x3), tmp32, None)
''', device_str='cuda')


# kernel path: inductor_cache/gm/cgmhcsqynipkfgmy3flzhdcpvxjl6eirmlrfdghyn5az7ssc5mx3.py
# Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   input_149 => mean_14
# Graph fragment:
#   %mean_14 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_84, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_28 = async_compile.triton('triton_poi_fused_mean_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/pl/cplotie3fqxijq27a6xyqrndqeybvvdxhjajwekil76g7vxd6vr4.py
# Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_159 => add_199, mul_291, mul_292, sub_92
#   x_160 => relu_87
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_737), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_292 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_291, %unsqueeze_741), kwargs = {})
#   %add_199 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_292, %unsqueeze_743), kwargs = {})
#   %relu_87 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_199,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 1024)
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


# kernel path: inductor_cache/ft/cftlg2qxbt3nj5dmrzztw4hi67mpctu6niovjcxxxwuttqsg3aw5.py
# Topologically Sorted Source Nodes: [x_166, x_167, add_14, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
# Source node to ATen node mapping:
#   add_14 => add_204
#   x_166 => add_203, mul_298, mul_299, sub_94
#   x_167 => relu_89
#   x_168 => relu_90
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %mul_298 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_298, %unsqueeze_757), kwargs = {})
#   %add_203 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_299, %unsqueeze_759), kwargs = {})
#   %relu_89 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_203,), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_84, %relu_89), kwargs = {})
#   %relu_90 : [num_users=4] = call_function[target=torch.ops.aten.relu.default](args = (%add_204,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 2048)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: inductor_cache/sr/csrisxzecfyazaw6ju2xwssjcn5fufxo7lwin3sac5oty3ox22ga.py
# Topologically Sorted Source Nodes: [x_177, x_178, add_15, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.threshold_backward]
# Source node to ATen node mapping:
#   add_15 => add_217
#   x_177 => add_216, mul_317, mul_318, sub_100
#   x_178 => relu_95
#   x_179 => relu_96
# Graph fragment:
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_801), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_317, %unsqueeze_805), kwargs = {})
#   %add_216 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %unsqueeze_807), kwargs = {})
#   %relu_95 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_216,), kwargs = {})
#   %add_217 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_90, %relu_95), kwargs = {})
#   %relu_96 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_217,), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu_96, 0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4) % 2048)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tl.full([1], 0, tl.int32)
    tmp18 = triton_helpers.maximum(tmp17, tmp16)
    tmp19 = tmp0 + tmp18
    tmp20 = triton_helpers.maximum(tmp17, tmp19)
    tmp21 = 0.0
    tmp22 = tmp20 <= tmp21
    tl.store(out_ptr0 + (x3), tmp20, None)
    tl.store(out_ptr1 + (x3), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/gb/cgb2c5o3ktor3vf73cwwzmnbibar6n526p7jsbttsy5xohof7rlb.py
# Topologically Sorted Source Nodes: [input_170, input_171, input_173, input_174, add_16, v], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_16 => add_222
#   input_170 => add_219, mul_320, mul_321, sub_101
#   input_171 => relu_97
#   input_173 => add_221, mul_323, mul_324, sub_102
#   input_174 => relu_98
#   v => mul_325
# Graph fragment:
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_101, %unsqueeze_809), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_101, %unsqueeze_811), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_320, %unsqueeze_813), kwargs = {})
#   %add_219 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_321, %unsqueeze_815), kwargs = {})
#   %relu_97 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_219,), kwargs = {})
#   %sub_102 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_102, %unsqueeze_817), kwargs = {})
#   %mul_323 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_102, %unsqueeze_819), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_323, %unsqueeze_821), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_324, %unsqueeze_823), kwargs = {})
#   %relu_98 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_221,), kwargs = {})
#   %add_222 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu_97, %relu_98), kwargs = {})
#   %mul_325 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_222, 0.5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x2), None)
    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp20 = tmp18 - tmp19
    tmp22 = tmp21 + tmp4
    tmp23 = libdevice.sqrt(tmp22)
    tmp24 = tmp7 / tmp23
    tmp25 = tmp24 * tmp9
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = triton_helpers.maximum(tmp16, tmp30)
    tmp32 = tmp17 + tmp31
    tmp33 = 0.5
    tmp34 = tmp32 * tmp33
    tl.store(in_out_ptr0 + (x2), tmp34, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (32, ), (1, ))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_42, (256, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_59, (32, ), (1, ))
    assert_size_stride(primals_60, (32, ), (1, ))
    assert_size_stride(primals_61, (32, ), (1, ))
    assert_size_stride(primals_62, (32, ), (1, ))
    assert_size_stride(primals_63, (32, ), (1, ))
    assert_size_stride(primals_64, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (128, ), (1, ))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (128, ), (1, ))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (128, ), (1, ))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (64, ), (1, ))
    assert_size_stride(primals_88, (64, ), (1, ))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_92, (32, ), (1, ))
    assert_size_stride(primals_93, (32, ), (1, ))
    assert_size_stride(primals_94, (32, ), (1, ))
    assert_size_stride(primals_95, (32, ), (1, ))
    assert_size_stride(primals_96, (32, ), (1, ))
    assert_size_stride(primals_97, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (128, ), (1, ))
    assert_size_stride(primals_99, (128, ), (1, ))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_103, (128, ), (1, ))
    assert_size_stride(primals_104, (128, ), (1, ))
    assert_size_stride(primals_105, (128, ), (1, ))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, ), (1, ))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_125, (32, ), (1, ))
    assert_size_stride(primals_126, (32, ), (1, ))
    assert_size_stride(primals_127, (32, ), (1, ))
    assert_size_stride(primals_128, (32, ), (1, ))
    assert_size_stride(primals_129, (32, ), (1, ))
    assert_size_stride(primals_130, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (256, ), (1, ))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_134, (256, ), (1, ))
    assert_size_stride(primals_135, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (256, ), (1, ))
    assert_size_stride(primals_139, (256, ), (1, ))
    assert_size_stride(primals_140, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_141, (512, ), (1, ))
    assert_size_stride(primals_142, (512, ), (1, ))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_151, (256, ), (1, ))
    assert_size_stride(primals_152, (256, ), (1, ))
    assert_size_stride(primals_153, (256, ), (1, ))
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_156, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_163, (32, ), (1, ))
    assert_size_stride(primals_164, (32, ), (1, ))
    assert_size_stride(primals_165, (32, ), (1, ))
    assert_size_stride(primals_166, (32, ), (1, ))
    assert_size_stride(primals_167, (32, ), (1, ))
    assert_size_stride(primals_168, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_169, (256, ), (1, ))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (256, ), (1, ))
    assert_size_stride(primals_177, (256, ), (1, ))
    assert_size_stride(primals_178, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (512, ), (1, ))
    assert_size_stride(primals_182, (512, ), (1, ))
    assert_size_stride(primals_183, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (256, ), (1, ))
    assert_size_stride(primals_186, (256, ), (1, ))
    assert_size_stride(primals_187, (256, ), (1, ))
    assert_size_stride(primals_188, (256, ), (1, ))
    assert_size_stride(primals_189, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_191, (128, ), (1, ))
    assert_size_stride(primals_192, (128, ), (1, ))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (128, ), (1, ))
    assert_size_stride(primals_195, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_196, (32, ), (1, ))
    assert_size_stride(primals_197, (32, ), (1, ))
    assert_size_stride(primals_198, (32, ), (1, ))
    assert_size_stride(primals_199, (32, ), (1, ))
    assert_size_stride(primals_200, (32, ), (1, ))
    assert_size_stride(primals_201, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_202, (256, ), (1, ))
    assert_size_stride(primals_203, (256, ), (1, ))
    assert_size_stride(primals_204, (256, ), (1, ))
    assert_size_stride(primals_205, (256, ), (1, ))
    assert_size_stride(primals_206, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (256, ), (1, ))
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_212, (512, ), (1, ))
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_217, (256, ), (1, ))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (256, ), (1, ))
    assert_size_stride(primals_222, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (128, ), (1, ))
    assert_size_stride(primals_225, (128, ), (1, ))
    assert_size_stride(primals_226, (128, ), (1, ))
    assert_size_stride(primals_227, (128, ), (1, ))
    assert_size_stride(primals_228, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_229, (32, ), (1, ))
    assert_size_stride(primals_230, (32, ), (1, ))
    assert_size_stride(primals_231, (32, ), (1, ))
    assert_size_stride(primals_232, (32, ), (1, ))
    assert_size_stride(primals_233, (32, ), (1, ))
    assert_size_stride(primals_234, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, ), (1, ))
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_240, (256, ), (1, ))
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (256, ), (1, ))
    assert_size_stride(primals_244, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_245, (512, ), (1, ))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (512, ), (1, ))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_249, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (512, ), (1, ))
    assert_size_stride(primals_252, (512, ), (1, ))
    assert_size_stride(primals_253, (512, ), (1, ))
    assert_size_stride(primals_254, (512, ), (1, ))
    assert_size_stride(primals_255, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (128, ), (1, ))
    assert_size_stride(primals_259, (128, ), (1, ))
    assert_size_stride(primals_260, (128, ), (1, ))
    assert_size_stride(primals_261, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_262, (32, ), (1, ))
    assert_size_stride(primals_263, (32, ), (1, ))
    assert_size_stride(primals_264, (32, ), (1, ))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, ), (1, ))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (512, ), (1, ))
    assert_size_stride(primals_272, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_278, (1024, ), (1, ))
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_283, (1024, ), (1, ))
    assert_size_stride(primals_284, (1024, ), (1, ))
    assert_size_stride(primals_285, (1024, ), (1, ))
    assert_size_stride(primals_286, (1024, ), (1, ))
    assert_size_stride(primals_287, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (512, ), (1, ))
    assert_size_stride(primals_290, (512, ), (1, ))
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_294, (128, ), (1, ))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_300, (32, ), (1, ))
    assert_size_stride(primals_301, (32, ), (1, ))
    assert_size_stride(primals_302, (32, ), (1, ))
    assert_size_stride(primals_303, (32, ), (1, ))
    assert_size_stride(primals_304, (32, ), (1, ))
    assert_size_stride(primals_305, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (512, ), (1, ))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_309, (512, ), (1, ))
    assert_size_stride(primals_310, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (512, ), (1, ))
    assert_size_stride(primals_314, (512, ), (1, ))
    assert_size_stride(primals_315, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_316, (1024, ), (1, ))
    assert_size_stride(primals_317, (1024, ), (1, ))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_319, (1024, ), (1, ))
    assert_size_stride(primals_320, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_321, (512, ), (1, ))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (512, ), (1, ))
    assert_size_stride(primals_324, (512, ), (1, ))
    assert_size_stride(primals_325, (512, ), (1, ))
    assert_size_stride(primals_326, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_327, (128, ), (1, ))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (128, ), (1, ))
    assert_size_stride(primals_331, (128, ), (1, ))
    assert_size_stride(primals_332, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_333, (32, ), (1, ))
    assert_size_stride(primals_334, (32, ), (1, ))
    assert_size_stride(primals_335, (32, ), (1, ))
    assert_size_stride(primals_336, (32, ), (1, ))
    assert_size_stride(primals_337, (32, ), (1, ))
    assert_size_stride(primals_338, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_339, (512, ), (1, ))
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (512, ), (1, ))
    assert_size_stride(primals_342, (512, ), (1, ))
    assert_size_stride(primals_343, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_344, (512, ), (1, ))
    assert_size_stride(primals_345, (512, ), (1, ))
    assert_size_stride(primals_346, (512, ), (1, ))
    assert_size_stride(primals_347, (512, ), (1, ))
    assert_size_stride(primals_348, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_349, (1024, ), (1, ))
    assert_size_stride(primals_350, (1024, ), (1, ))
    assert_size_stride(primals_351, (1024, ), (1, ))
    assert_size_stride(primals_352, (1024, ), (1, ))
    assert_size_stride(primals_353, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (512, ), (1, ))
    assert_size_stride(primals_356, (512, ), (1, ))
    assert_size_stride(primals_357, (512, ), (1, ))
    assert_size_stride(primals_358, (512, ), (1, ))
    assert_size_stride(primals_359, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_360, (128, ), (1, ))
    assert_size_stride(primals_361, (128, ), (1, ))
    assert_size_stride(primals_362, (128, ), (1, ))
    assert_size_stride(primals_363, (128, ), (1, ))
    assert_size_stride(primals_364, (128, ), (1, ))
    assert_size_stride(primals_365, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_366, (32, ), (1, ))
    assert_size_stride(primals_367, (32, ), (1, ))
    assert_size_stride(primals_368, (32, ), (1, ))
    assert_size_stride(primals_369, (32, ), (1, ))
    assert_size_stride(primals_370, (32, ), (1, ))
    assert_size_stride(primals_371, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_372, (512, ), (1, ))
    assert_size_stride(primals_373, (512, ), (1, ))
    assert_size_stride(primals_374, (512, ), (1, ))
    assert_size_stride(primals_375, (512, ), (1, ))
    assert_size_stride(primals_376, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_377, (512, ), (1, ))
    assert_size_stride(primals_378, (512, ), (1, ))
    assert_size_stride(primals_379, (512, ), (1, ))
    assert_size_stride(primals_380, (512, ), (1, ))
    assert_size_stride(primals_381, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_383, (1024, ), (1, ))
    assert_size_stride(primals_384, (1024, ), (1, ))
    assert_size_stride(primals_385, (1024, ), (1, ))
    assert_size_stride(primals_386, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_387, (512, ), (1, ))
    assert_size_stride(primals_388, (512, ), (1, ))
    assert_size_stride(primals_389, (512, ), (1, ))
    assert_size_stride(primals_390, (512, ), (1, ))
    assert_size_stride(primals_391, (512, ), (1, ))
    assert_size_stride(primals_392, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_393, (128, ), (1, ))
    assert_size_stride(primals_394, (128, ), (1, ))
    assert_size_stride(primals_395, (128, ), (1, ))
    assert_size_stride(primals_396, (128, ), (1, ))
    assert_size_stride(primals_397, (128, ), (1, ))
    assert_size_stride(primals_398, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_399, (32, ), (1, ))
    assert_size_stride(primals_400, (32, ), (1, ))
    assert_size_stride(primals_401, (32, ), (1, ))
    assert_size_stride(primals_402, (32, ), (1, ))
    assert_size_stride(primals_403, (32, ), (1, ))
    assert_size_stride(primals_404, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (512, ), (1, ))
    assert_size_stride(primals_407, (512, ), (1, ))
    assert_size_stride(primals_408, (512, ), (1, ))
    assert_size_stride(primals_409, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_410, (512, ), (1, ))
    assert_size_stride(primals_411, (512, ), (1, ))
    assert_size_stride(primals_412, (512, ), (1, ))
    assert_size_stride(primals_413, (512, ), (1, ))
    assert_size_stride(primals_414, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_415, (1024, ), (1, ))
    assert_size_stride(primals_416, (1024, ), (1, ))
    assert_size_stride(primals_417, (1024, ), (1, ))
    assert_size_stride(primals_418, (1024, ), (1, ))
    assert_size_stride(primals_419, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_420, (512, ), (1, ))
    assert_size_stride(primals_421, (512, ), (1, ))
    assert_size_stride(primals_422, (512, ), (1, ))
    assert_size_stride(primals_423, (512, ), (1, ))
    assert_size_stride(primals_424, (512, ), (1, ))
    assert_size_stride(primals_425, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_426, (128, ), (1, ))
    assert_size_stride(primals_427, (128, ), (1, ))
    assert_size_stride(primals_428, (128, ), (1, ))
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (128, ), (1, ))
    assert_size_stride(primals_431, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_432, (32, ), (1, ))
    assert_size_stride(primals_433, (32, ), (1, ))
    assert_size_stride(primals_434, (32, ), (1, ))
    assert_size_stride(primals_435, (32, ), (1, ))
    assert_size_stride(primals_436, (32, ), (1, ))
    assert_size_stride(primals_437, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_438, (512, ), (1, ))
    assert_size_stride(primals_439, (512, ), (1, ))
    assert_size_stride(primals_440, (512, ), (1, ))
    assert_size_stride(primals_441, (512, ), (1, ))
    assert_size_stride(primals_442, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_443, (512, ), (1, ))
    assert_size_stride(primals_444, (512, ), (1, ))
    assert_size_stride(primals_445, (512, ), (1, ))
    assert_size_stride(primals_446, (512, ), (1, ))
    assert_size_stride(primals_447, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_448, (1024, ), (1, ))
    assert_size_stride(primals_449, (1024, ), (1, ))
    assert_size_stride(primals_450, (1024, ), (1, ))
    assert_size_stride(primals_451, (1024, ), (1, ))
    assert_size_stride(primals_452, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_453, (512, ), (1, ))
    assert_size_stride(primals_454, (512, ), (1, ))
    assert_size_stride(primals_455, (512, ), (1, ))
    assert_size_stride(primals_456, (512, ), (1, ))
    assert_size_stride(primals_457, (512, ), (1, ))
    assert_size_stride(primals_458, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (128, ), (1, ))
    assert_size_stride(primals_461, (128, ), (1, ))
    assert_size_stride(primals_462, (128, ), (1, ))
    assert_size_stride(primals_463, (128, ), (1, ))
    assert_size_stride(primals_464, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_465, (32, ), (1, ))
    assert_size_stride(primals_466, (32, ), (1, ))
    assert_size_stride(primals_467, (32, ), (1, ))
    assert_size_stride(primals_468, (32, ), (1, ))
    assert_size_stride(primals_469, (32, ), (1, ))
    assert_size_stride(primals_470, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_471, (1024, ), (1, ))
    assert_size_stride(primals_472, (1024, ), (1, ))
    assert_size_stride(primals_473, (1024, ), (1, ))
    assert_size_stride(primals_474, (1024, ), (1, ))
    assert_size_stride(primals_475, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_476, (1024, ), (1, ))
    assert_size_stride(primals_477, (1024, ), (1, ))
    assert_size_stride(primals_478, (1024, ), (1, ))
    assert_size_stride(primals_479, (1024, ), (1, ))
    assert_size_stride(primals_480, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_481, (2048, ), (1, ))
    assert_size_stride(primals_482, (2048, ), (1, ))
    assert_size_stride(primals_483, (2048, ), (1, ))
    assert_size_stride(primals_484, (2048, ), (1, ))
    assert_size_stride(primals_485, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_486, (2048, ), (1, ))
    assert_size_stride(primals_487, (2048, ), (1, ))
    assert_size_stride(primals_488, (2048, ), (1, ))
    assert_size_stride(primals_489, (2048, ), (1, ))
    assert_size_stride(primals_490, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_491, (512, ), (1, ))
    assert_size_stride(primals_492, (512, ), (1, ))
    assert_size_stride(primals_493, (512, ), (1, ))
    assert_size_stride(primals_494, (512, ), (1, ))
    assert_size_stride(primals_495, (512, ), (1, ))
    assert_size_stride(primals_496, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_497, (128, ), (1, ))
    assert_size_stride(primals_498, (128, ), (1, ))
    assert_size_stride(primals_499, (128, ), (1, ))
    assert_size_stride(primals_500, (128, ), (1, ))
    assert_size_stride(primals_501, (128, ), (1, ))
    assert_size_stride(primals_502, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_503, (32, ), (1, ))
    assert_size_stride(primals_504, (32, ), (1, ))
    assert_size_stride(primals_505, (32, ), (1, ))
    assert_size_stride(primals_506, (32, ), (1, ))
    assert_size_stride(primals_507, (32, ), (1, ))
    assert_size_stride(primals_508, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_509, (1024, ), (1, ))
    assert_size_stride(primals_510, (1024, ), (1, ))
    assert_size_stride(primals_511, (1024, ), (1, ))
    assert_size_stride(primals_512, (1024, ), (1, ))
    assert_size_stride(primals_513, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_514, (1024, ), (1, ))
    assert_size_stride(primals_515, (1024, ), (1, ))
    assert_size_stride(primals_516, (1024, ), (1, ))
    assert_size_stride(primals_517, (1024, ), (1, ))
    assert_size_stride(primals_518, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_519, (2048, ), (1, ))
    assert_size_stride(primals_520, (2048, ), (1, ))
    assert_size_stride(primals_521, (2048, ), (1, ))
    assert_size_stride(primals_522, (2048, ), (1, ))
    assert_size_stride(primals_523, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_524, (512, ), (1, ))
    assert_size_stride(primals_525, (512, ), (1, ))
    assert_size_stride(primals_526, (512, ), (1, ))
    assert_size_stride(primals_527, (512, ), (1, ))
    assert_size_stride(primals_528, (512, ), (1, ))
    assert_size_stride(primals_529, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_530, (128, ), (1, ))
    assert_size_stride(primals_531, (128, ), (1, ))
    assert_size_stride(primals_532, (128, ), (1, ))
    assert_size_stride(primals_533, (128, ), (1, ))
    assert_size_stride(primals_534, (128, ), (1, ))
    assert_size_stride(primals_535, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_536, (32, ), (1, ))
    assert_size_stride(primals_537, (32, ), (1, ))
    assert_size_stride(primals_538, (32, ), (1, ))
    assert_size_stride(primals_539, (32, ), (1, ))
    assert_size_stride(primals_540, (32, ), (1, ))
    assert_size_stride(primals_541, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_542, (1024, ), (1, ))
    assert_size_stride(primals_543, (1024, ), (1, ))
    assert_size_stride(primals_544, (1024, ), (1, ))
    assert_size_stride(primals_545, (1024, ), (1, ))
    assert_size_stride(primals_546, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_547, (1024, ), (1, ))
    assert_size_stride(primals_548, (1024, ), (1, ))
    assert_size_stride(primals_549, (1024, ), (1, ))
    assert_size_stride(primals_550, (1024, ), (1, ))
    assert_size_stride(primals_551, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_552, (2048, ), (1, ))
    assert_size_stride(primals_553, (2048, ), (1, ))
    assert_size_stride(primals_554, (2048, ), (1, ))
    assert_size_stride(primals_555, (2048, ), (1, ))
    assert_size_stride(primals_556, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_557, (1024, ), (1, ))
    assert_size_stride(primals_558, (1024, ), (1, ))
    assert_size_stride(primals_559, (1024, ), (1, ))
    assert_size_stride(primals_560, (1024, ), (1, ))
    assert_size_stride(primals_561, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_562, (1024, ), (1, ))
    assert_size_stride(primals_563, (1024, ), (1, ))
    assert_size_stride(primals_564, (1024, ), (1, ))
    assert_size_stride(primals_565, (1024, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 32, 32), (65536, 1024, 32, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 64, 32, 32), (65536, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, x_1, x_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, primals_2, primals_4, primals_5, primals_6, primals_7, buf2, 262144, grid=grid(262144), stream=stream0)
        del primals_2
        del primals_7
        buf3 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 64, 16, 16), (16384, 256, 16, 1), torch.int8)
        buf5 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 256, 256), torch.float32)
        buf6 = reinterpret_tensor(buf5, (4, 64, 1, 1), (64, 1, 1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [x_3, input_1], Original ATen: [aten.max_pool2d_with_indices, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_max_pool2d_with_indices_mean_1.run(buf6, buf2, buf3, buf4, 256, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 128, 1, 1), (128, 1, 1, 1))
        buf8 = buf7; del buf7  # reuse
        buf9 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3, input_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf8, primals_9, primals_10, primals_11, primals_12, primals_13, buf9, 512, grid=grid(512), stream=stream0)
        del primals_13
        del primals_9
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 64, 1, 1), (64, 1, 1, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6, input_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf11, primals_15, primals_16, primals_17, primals_18, primals_19, buf12, 256, grid=grid(256), stream=stream0)
        del primals_15
        del primals_19
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 32, 1, 1), (32, 1, 1, 1))
        buf14 = buf13; del buf13  # reuse
        buf299 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        buf15 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_8, input_9, input_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf14, primals_21, primals_22, primals_23, primals_24, primals_25, buf15, 128, grid=grid(128), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf3, primals_26, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf17 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf16, primals_27, primals_28, primals_29, primals_30, buf17, 131072, grid=grid(131072), stream=stream0)
        del primals_30
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf18, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf19 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_6.run(buf15, buf19, 512, grid=grid(512), stream=stream0)
        buf20 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8, x_9, x_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_7.run(buf19, buf18, primals_32, primals_33, primals_34, primals_35, buf20, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 256, 16, 16), (65536, 256, 16, 1))
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf3, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf23 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf26 = reinterpret_tensor(buf25, (4, 256, 1, 1), (256, 1, 1, 1), 0); del buf25  # reuse
        # Topologically Sorted Source Nodes: [x_12, x_13, input_12, add, x_14, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_8.run(buf24, buf26, buf22, primals_42, primals_43, primals_44, primals_45, buf21, primals_37, primals_38, primals_39, primals_40, 1024, 256, grid=grid(1024), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 128, 1, 1), (128, 1, 1, 1))
        buf28 = buf27; del buf27  # reuse
        buf29 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15, input_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf28, primals_47, primals_48, primals_49, primals_50, primals_51, buf29, 512, grid=grid(512), stream=stream0)
        del primals_47
        del primals_51
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 64, 1, 1), (64, 1, 1, 1))
        buf31 = buf30; del buf30  # reuse
        buf32 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17, input_18, input_19], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf31, primals_53, primals_54, primals_55, primals_56, primals_57, buf32, 256, grid=grid(256), stream=stream0)
        del primals_53
        del primals_57
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 32, 1, 1), (32, 1, 1, 1))
        buf34 = buf33; del buf33  # reuse
        buf35 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 32)  # alias
        # Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf34, primals_59, primals_60, primals_61, primals_62, primals_63, buf35, 128, grid=grid(128), stream=stream0)
        del primals_59
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf24, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf37 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf36, primals_65, primals_66, primals_67, primals_68, buf37, 131072, grid=grid(131072), stream=stream0)
        del primals_68
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf38, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf39 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_4], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_6.run(buf35, buf39, 512, grid=grid(512), stream=stream0)
        buf40 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_7.run(buf39, buf38, primals_70, primals_71, primals_72, primals_73, buf40, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf42 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf43 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf44 = reinterpret_tensor(buf43, (4, 256, 1, 1), (256, 1, 1, 1), 0); del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_23, x_24, add_1, x_25, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_9.run(buf44, buf24, buf41, primals_75, primals_76, primals_77, primals_78, buf42, 1024, 256, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 128, 1, 1), (128, 1, 1, 1))
        buf46 = buf45; del buf45  # reuse
        buf47 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25, input_26], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf46, primals_80, primals_81, primals_82, primals_83, primals_84, buf47, 512, grid=grid(512), stream=stream0)
        del primals_80
        del primals_84
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 1, 1), (64, 1, 1, 1))
        buf49 = buf48; del buf48  # reuse
        buf50 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_27, input_28, input_29], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(buf49, primals_86, primals_87, primals_88, primals_89, primals_90, buf50, 256, grid=grid(256), stream=stream0)
        del primals_86
        del primals_90
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 32, 1, 1), (32, 1, 1, 1))
        buf52 = buf51; del buf51  # reuse
        buf53 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 64)  # alias
        # Topologically Sorted Source Nodes: [input_30, input_31, input_32], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf52, primals_92, primals_93, primals_94, primals_95, primals_96, buf53, 128, grid=grid(128), stream=stream0)
        del primals_92
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf42, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf55 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf54, primals_98, primals_99, primals_100, primals_101, buf55, 131072, grid=grid(131072), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf56, (4, 128, 16, 16), (32768, 256, 16, 1))
        buf57 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_8], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_6.run(buf53, buf57, 512, grid=grid(512), stream=stream0)
        buf58 = empty_strided_cuda((4, 128, 16, 16), (32768, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_30, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_7.run(buf57, buf56, primals_103, primals_104, primals_105, primals_106, buf58, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf60 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        buf61 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf62 = reinterpret_tensor(buf61, (4, 256, 1, 1), (256, 1, 1, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_34, x_35, add_2, x_36, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_9.run(buf62, buf42, buf59, primals_108, primals_109, primals_110, primals_111, buf60, 1024, 256, grid=grid(1024), stream=stream0)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 256, 1, 1), (256, 1, 1, 1))
        buf64 = buf63; del buf63  # reuse
        buf65 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf64, primals_113, primals_114, primals_115, primals_116, primals_117, buf65, 1024, grid=grid(1024), stream=stream0)
        del primals_113
        del primals_117
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 128, 1, 1), (128, 1, 1, 1))
        buf67 = buf66; del buf66  # reuse
        buf68 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37, input_38, input_39], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf67, primals_119, primals_120, primals_121, primals_122, primals_123, buf68, 512, grid=grid(512), stream=stream0)
        del primals_119
        del primals_123
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 32, 1, 1), (32, 1, 1, 1))
        buf70 = buf69; del buf69  # reuse
        buf71 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 96)  # alias
        # Topologically Sorted Source Nodes: [input_40, input_41, input_42], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf70, primals_125, primals_126, primals_127, primals_128, primals_129, buf71, 128, grid=grid(128), stream=stream0)
        del primals_125
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf60, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 256, 16, 16), (65536, 256, 16, 1))
        buf73 = empty_strided_cuda((4, 256, 16, 16), (65536, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_38, x_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf72, primals_131, primals_132, primals_133, primals_134, buf73, 262144, grid=grid(262144), stream=stream0)
        del primals_134
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_135, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf74, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf75 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_12], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_12.run(buf71, buf75, 1024, grid=grid(1024), stream=stream0)
        buf76 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_41, x_42, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13.run(buf75, buf74, primals_136, primals_137, primals_138, primals_139, buf76, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 512, 8, 8), (32768, 64, 8, 1))
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf60, primals_145, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf79 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf80 = buf79; del buf79  # reuse
        buf81 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf82 = reinterpret_tensor(buf81, (4, 512, 1, 1), (512, 1, 1, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_45, x_46, input_44, add_3, x_47, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_14.run(buf80, buf82, buf78, primals_146, primals_147, primals_148, primals_149, buf77, primals_141, primals_142, primals_143, primals_144, 2048, 64, grid=grid(2048), stream=stream0)
        del primals_149
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 256, 1, 1), (256, 1, 1, 1))
        buf84 = buf83; del buf83  # reuse
        buf85 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, input_47, input_48], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf84, primals_151, primals_152, primals_153, primals_154, primals_155, buf85, 1024, grid=grid(1024), stream=stream0)
        del primals_151
        del primals_155
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 128, 1, 1), (128, 1, 1, 1))
        buf87 = buf86; del buf86  # reuse
        buf88 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_49, input_50, input_51], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf87, primals_157, primals_158, primals_159, primals_160, primals_161, buf88, 512, grid=grid(512), stream=stream0)
        del primals_157
        del primals_161
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 32, 1, 1), (32, 1, 1, 1))
        buf90 = buf89; del buf89  # reuse
        buf91 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [input_52, input_53, input_54], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf90, primals_163, primals_164, primals_165, primals_166, primals_167, buf91, 128, grid=grid(128), stream=stream0)
        del primals_163
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf80, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf93 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf92, primals_169, primals_170, primals_171, primals_172, buf93, 65536, grid=grid(65536), stream=stream0)
        del primals_172
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_173, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf94, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf95 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_16], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_12.run(buf91, buf95, 1024, grid=grid(1024), stream=stream0)
        buf96 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_52, x_53, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13.run(buf95, buf94, primals_174, primals_175, primals_176, primals_177, buf96, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf98 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf99 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf100 = reinterpret_tensor(buf99, (4, 512, 1, 1), (512, 1, 1, 1), 0); del buf99  # reuse
        # Topologically Sorted Source Nodes: [x_56, x_57, add_4, x_58, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_16.run(buf100, buf80, buf97, primals_179, primals_180, primals_181, primals_182, buf98, 2048, 64, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 256, 1, 1), (256, 1, 1, 1))
        buf102 = buf101; del buf101  # reuse
        buf103 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, input_57, input_58], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf102, primals_184, primals_185, primals_186, primals_187, primals_188, buf103, 1024, grid=grid(1024), stream=stream0)
        del primals_184
        del primals_188
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 128, 1, 1), (128, 1, 1, 1))
        buf105 = buf104; del buf104  # reuse
        buf106 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59, input_60, input_61], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf105, primals_190, primals_191, primals_192, primals_193, primals_194, buf106, 512, grid=grid(512), stream=stream0)
        del primals_190
        del primals_194
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 32, 1, 1), (32, 1, 1, 1))
        buf108 = buf107; del buf107  # reuse
        buf109 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 160)  # alias
        # Topologically Sorted Source Nodes: [input_62, input_63, input_64], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf108, primals_196, primals_197, primals_198, primals_199, primals_200, buf109, 128, grid=grid(128), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf98, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf111 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf110, primals_202, primals_203, primals_204, primals_205, buf111, 65536, grid=grid(65536), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_206, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf112, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf113 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_20], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_12.run(buf109, buf113, 1024, grid=grid(1024), stream=stream0)
        buf114 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_63, x_64, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13.run(buf113, buf112, primals_207, primals_208, primals_209, primals_210, buf114, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf116 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf117 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf118 = reinterpret_tensor(buf117, (4, 512, 1, 1), (512, 1, 1, 1), 0); del buf117  # reuse
        # Topologically Sorted Source Nodes: [x_67, x_68, add_5, x_69, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_16.run(buf118, buf98, buf115, primals_212, primals_213, primals_214, primals_215, buf116, 2048, 64, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 256, 1, 1), (256, 1, 1, 1))
        buf120 = buf119; del buf119  # reuse
        buf121 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_66, input_67, input_68], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf120, primals_217, primals_218, primals_219, primals_220, primals_221, buf121, 1024, grid=grid(1024), stream=stream0)
        del primals_217
        del primals_221
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 128, 1, 1), (128, 1, 1, 1))
        buf123 = buf122; del buf122  # reuse
        buf124 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_69, input_70, input_71], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf123, primals_223, primals_224, primals_225, primals_226, primals_227, buf124, 512, grid=grid(512), stream=stream0)
        del primals_223
        del primals_227
        # Topologically Sorted Source Nodes: [input_72], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 32, 1, 1), (32, 1, 1, 1))
        buf126 = buf125; del buf125  # reuse
        buf127 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 192)  # alias
        # Topologically Sorted Source Nodes: [input_72, input_73, input_74], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf126, primals_229, primals_230, primals_231, primals_232, primals_233, buf127, 128, grid=grid(128), stream=stream0)
        del primals_229
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf116, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf129 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_71, x_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_15.run(buf128, primals_235, primals_236, primals_237, primals_238, buf129, 65536, grid=grid(65536), stream=stream0)
        del primals_238
        # Topologically Sorted Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf130, (4, 256, 8, 8), (16384, 64, 8, 1))
        buf131 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_24], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_12.run(buf127, buf131, 1024, grid=grid(1024), stream=stream0)
        buf132 = empty_strided_cuda((4, 256, 8, 8), (16384, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_74, x_75, x_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_13.run(buf131, buf130, primals_240, primals_241, primals_242, primals_243, buf132, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf134 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        buf135 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf136 = reinterpret_tensor(buf135, (4, 512, 1, 1), (512, 1, 1, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [x_78, x_79, add_6, x_80, input_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_16.run(buf136, buf116, buf133, primals_245, primals_246, primals_247, primals_248, buf134, 2048, 64, grid=grid(2048), stream=stream0)
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 512, 1, 1), (512, 1, 1, 1))
        buf138 = buf137; del buf137  # reuse
        buf139 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, input_77, input_78], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf138, primals_250, primals_251, primals_252, primals_253, primals_254, buf139, 2048, grid=grid(2048), stream=stream0)
        del primals_250
        del primals_254
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_255, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 128, 1, 1), (128, 1, 1, 1))
        buf141 = buf140; del buf140  # reuse
        buf142 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_79, input_80, input_81], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf141, primals_256, primals_257, primals_258, primals_259, primals_260, buf142, 512, grid=grid(512), stream=stream0)
        del primals_256
        del primals_260
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 32, 1, 1), (32, 1, 1, 1))
        buf144 = buf143; del buf143  # reuse
        buf145 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 224)  # alias
        # Topologically Sorted Source Nodes: [input_82, input_83, input_84], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf144, primals_262, primals_263, primals_264, primals_265, primals_266, buf145, 128, grid=grid(128), stream=stream0)
        del primals_262
        # Topologically Sorted Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf134, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 512, 8, 8), (32768, 64, 8, 1))
        buf147 = empty_strided_cuda((4, 512, 8, 8), (32768, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_82, x_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf146, primals_268, primals_269, primals_270, primals_271, buf147, 131072, grid=grid(131072), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_272, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf148, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf149 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_28], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_19.run(buf145, buf149, 2048, grid=grid(2048), stream=stream0)
        buf150 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_85, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20.run(buf149, buf148, primals_273, primals_274, primals_275, primals_276, buf150, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 1024, 4, 4), (16384, 16, 4, 1))
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf134, primals_282, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf153 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf154 = buf153; del buf153  # reuse
        buf155 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf156 = reinterpret_tensor(buf155, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_89, x_90, input_86, add_7, x_91, input_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_21.run(buf154, buf156, buf152, primals_283, primals_284, primals_285, primals_286, buf151, primals_278, primals_279, primals_280, primals_281, 4096, 16, grid=grid(4096), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 512, 1, 1), (512, 1, 1, 1))
        buf158 = buf157; del buf157  # reuse
        buf159 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_88, input_89, input_90], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf158, primals_288, primals_289, primals_290, primals_291, primals_292, buf159, 2048, grid=grid(2048), stream=stream0)
        del primals_288
        del primals_292
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_293, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 128, 1, 1), (128, 1, 1, 1))
        buf161 = buf160; del buf160  # reuse
        buf162 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_91, input_92, input_93], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf161, primals_294, primals_295, primals_296, primals_297, primals_298, buf162, 512, grid=grid(512), stream=stream0)
        del primals_294
        del primals_298
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 32, 1, 1), (32, 1, 1, 1))
        buf164 = buf163; del buf163  # reuse
        buf165 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 256)  # alias
        # Topologically Sorted Source Nodes: [input_94, input_95, input_96], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf164, primals_300, primals_301, primals_302, primals_303, primals_304, buf165, 128, grid=grid(128), stream=stream0)
        del primals_300
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf154, primals_305, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf167 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_93, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf166, primals_306, primals_307, primals_308, primals_309, buf167, 32768, grid=grid(32768), stream=stream0)
        del primals_309
        # Topologically Sorted Source Nodes: [x_95], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_310, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf168, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf169 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_32], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_19.run(buf165, buf169, 2048, grid=grid(2048), stream=stream0)
        buf170 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_96, x_97, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20.run(buf169, buf168, primals_311, primals_312, primals_313, primals_314, buf170, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf172 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf173 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf174 = reinterpret_tensor(buf173, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf173  # reuse
        # Topologically Sorted Source Nodes: [x_100, x_101, add_8, x_102, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23.run(buf174, buf154, buf171, primals_316, primals_317, primals_318, primals_319, buf172, 4096, 16, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_98], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_320, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 512, 1, 1), (512, 1, 1, 1))
        buf176 = buf175; del buf175  # reuse
        buf177 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_98, input_99, input_100], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf176, primals_321, primals_322, primals_323, primals_324, primals_325, buf177, 2048, grid=grid(2048), stream=stream0)
        del primals_321
        del primals_325
        # Topologically Sorted Source Nodes: [input_101], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_326, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 128, 1, 1), (128, 1, 1, 1))
        buf179 = buf178; del buf178  # reuse
        buf180 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_101, input_102, input_103], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf179, primals_327, primals_328, primals_329, primals_330, primals_331, buf180, 512, grid=grid(512), stream=stream0)
        del primals_327
        del primals_331
        # Topologically Sorted Source Nodes: [input_104], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (4, 32, 1, 1), (32, 1, 1, 1))
        buf182 = buf181; del buf181  # reuse
        buf183 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 288)  # alias
        # Topologically Sorted Source Nodes: [input_104, input_105, input_106], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf182, primals_333, primals_334, primals_335, primals_336, primals_337, buf183, 128, grid=grid(128), stream=stream0)
        del primals_333
        # Topologically Sorted Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf172, primals_338, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf185 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_104, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf184, primals_339, primals_340, primals_341, primals_342, buf185, 32768, grid=grid(32768), stream=stream0)
        del primals_342
        # Topologically Sorted Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_343, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf186, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf187 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_36], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_19.run(buf183, buf187, 2048, grid=grid(2048), stream=stream0)
        buf188 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_107, x_108, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20.run(buf187, buf186, primals_344, primals_345, primals_346, primals_347, buf188, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_348, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf190 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf191 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf192 = reinterpret_tensor(buf191, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_111, x_112, add_9, x_113, input_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23.run(buf192, buf172, buf189, primals_349, primals_350, primals_351, primals_352, buf190, 4096, 16, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_108], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_353, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 512, 1, 1), (512, 1, 1, 1))
        buf194 = buf193; del buf193  # reuse
        buf195 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_108, input_109, input_110], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf194, primals_354, primals_355, primals_356, primals_357, primals_358, buf195, 2048, grid=grid(2048), stream=stream0)
        del primals_354
        del primals_358
        # Topologically Sorted Source Nodes: [input_111], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_359, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 128, 1, 1), (128, 1, 1, 1))
        buf197 = buf196; del buf196  # reuse
        buf198 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_111, input_112, input_113], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf197, primals_360, primals_361, primals_362, primals_363, primals_364, buf198, 512, grid=grid(512), stream=stream0)
        del primals_360
        del primals_364
        # Topologically Sorted Source Nodes: [input_114], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_365, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 32, 1, 1), (32, 1, 1, 1))
        buf200 = buf199; del buf199  # reuse
        buf201 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 320)  # alias
        # Topologically Sorted Source Nodes: [input_114, input_115, input_116], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf200, primals_366, primals_367, primals_368, primals_369, primals_370, buf201, 128, grid=grid(128), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf190, primals_371, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf203 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_115, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf202, primals_372, primals_373, primals_374, primals_375, buf203, 32768, grid=grid(32768), stream=stream0)
        del primals_375
        # Topologically Sorted Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_376, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf204, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf205 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_40], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_19.run(buf201, buf205, 2048, grid=grid(2048), stream=stream0)
        buf206 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_118, x_119, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20.run(buf205, buf204, primals_377, primals_378, primals_379, primals_380, buf206, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_381, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf208 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf209 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf210 = reinterpret_tensor(buf209, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_122, x_123, add_10, x_124, input_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23.run(buf210, buf190, buf207, primals_382, primals_383, primals_384, primals_385, buf208, 4096, 16, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_386, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 512, 1, 1), (512, 1, 1, 1))
        buf212 = buf211; del buf211  # reuse
        buf213 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_118, input_119, input_120], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf212, primals_387, primals_388, primals_389, primals_390, primals_391, buf213, 2048, grid=grid(2048), stream=stream0)
        del primals_387
        del primals_391
        # Topologically Sorted Source Nodes: [input_121], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 128, 1, 1), (128, 1, 1, 1))
        buf215 = buf214; del buf214  # reuse
        buf216 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_121, input_122, input_123], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf215, primals_393, primals_394, primals_395, primals_396, primals_397, buf216, 512, grid=grid(512), stream=stream0)
        del primals_393
        del primals_397
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_398, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 32, 1, 1), (32, 1, 1, 1))
        buf218 = buf217; del buf217  # reuse
        buf219 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 352)  # alias
        # Topologically Sorted Source Nodes: [input_124, input_125, input_126], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf218, primals_399, primals_400, primals_401, primals_402, primals_403, buf219, 128, grid=grid(128), stream=stream0)
        del primals_399
        # Topologically Sorted Source Nodes: [x_125], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf208, primals_404, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf221 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_126, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf220, primals_405, primals_406, primals_407, primals_408, buf221, 32768, grid=grid(32768), stream=stream0)
        del primals_408
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_409, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf222, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf223 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_44], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_19.run(buf219, buf223, 2048, grid=grid(2048), stream=stream0)
        buf224 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_129, x_130, x_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20.run(buf223, buf222, primals_410, primals_411, primals_412, primals_413, buf224, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_414, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf226 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf227 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf228 = reinterpret_tensor(buf227, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_133, x_134, add_11, x_135, input_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23.run(buf228, buf208, buf225, primals_415, primals_416, primals_417, primals_418, buf226, 4096, 16, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_419, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 512, 1, 1), (512, 1, 1, 1))
        buf230 = buf229; del buf229  # reuse
        buf231 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_128, input_129, input_130], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf230, primals_420, primals_421, primals_422, primals_423, primals_424, buf231, 2048, grid=grid(2048), stream=stream0)
        del primals_420
        del primals_424
        # Topologically Sorted Source Nodes: [input_131], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_425, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 128, 1, 1), (128, 1, 1, 1))
        buf233 = buf232; del buf232  # reuse
        buf234 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_131, input_132, input_133], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf233, primals_426, primals_427, primals_428, primals_429, primals_430, buf234, 512, grid=grid(512), stream=stream0)
        del primals_426
        del primals_430
        # Topologically Sorted Source Nodes: [input_134], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_431, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 32, 1, 1), (32, 1, 1, 1))
        buf236 = buf235; del buf235  # reuse
        buf237 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 384)  # alias
        # Topologically Sorted Source Nodes: [input_134, input_135, input_136], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf236, primals_432, primals_433, primals_434, primals_435, primals_436, buf237, 128, grid=grid(128), stream=stream0)
        del primals_432
        # Topologically Sorted Source Nodes: [x_136], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf226, primals_437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf239 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_137, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf238, primals_438, primals_439, primals_440, primals_441, buf239, 32768, grid=grid(32768), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf240, (4, 512, 4, 4), (8192, 16, 4, 1))
        buf241 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_48], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_19.run(buf237, buf241, 2048, grid=grid(2048), stream=stream0)
        buf242 = empty_strided_cuda((4, 512, 4, 4), (8192, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_140, x_141, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_20.run(buf241, buf240, primals_443, primals_444, primals_445, primals_446, buf242, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [x_143], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf244 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        buf245 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf246 = reinterpret_tensor(buf245, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [x_144, x_145, add_12, x_146, input_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_23.run(buf246, buf226, buf243, primals_448, primals_449, primals_450, primals_451, buf244, 4096, 16, grid=grid(4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_138], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 512, 1, 1), (512, 1, 1, 1))
        buf248 = buf247; del buf247  # reuse
        buf249 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_138, input_139, input_140], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf248, primals_453, primals_454, primals_455, primals_456, primals_457, buf249, 2048, grid=grid(2048), stream=stream0)
        del primals_453
        del primals_457
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_458, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 128, 1, 1), (128, 1, 1, 1))
        buf251 = buf250; del buf250  # reuse
        buf252 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_141, input_142, input_143], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf251, primals_459, primals_460, primals_461, primals_462, primals_463, buf252, 512, grid=grid(512), stream=stream0)
        del primals_459
        del primals_463
        # Topologically Sorted Source Nodes: [input_144], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_464, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 32, 1, 1), (32, 1, 1, 1))
        buf254 = buf253; del buf253  # reuse
        buf255 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 416)  # alias
        # Topologically Sorted Source Nodes: [input_144, input_145, input_146], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf254, primals_465, primals_466, primals_467, primals_468, primals_469, buf255, 128, grid=grid(128), stream=stream0)
        del primals_465
        # Topologically Sorted Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf244, primals_470, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 1024, 4, 4), (16384, 16, 4, 1))
        buf257 = empty_strided_cuda((4, 1024, 4, 4), (16384, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_148, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_24.run(buf256, primals_471, primals_472, primals_473, primals_474, buf257, 65536, grid=grid(65536), stream=stream0)
        del primals_474
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_475, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf258, (4, 1024, 2, 2), (4096, 4, 2, 1))
        buf259 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_52], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_25.run(buf255, buf259, 4096, grid=grid(4096), stream=stream0)
        buf260 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_151, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_26.run(buf259, buf258, primals_476, primals_477, primals_478, primals_479, buf260, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_480, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 2048, 2, 2), (8192, 4, 2, 1))
        # Topologically Sorted Source Nodes: [input_147], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf244, primals_485, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 2048, 2, 2), (8192, 4, 2, 1))
        buf263 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.float32)
        buf264 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [x_155, x_156, input_148, add_13, x_157], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_27.run(buf264, buf262, primals_486, primals_487, primals_488, primals_489, buf261, primals_481, primals_482, primals_483, primals_484, 32768, grid=grid(32768), stream=stream0)
        del primals_489
        buf265 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_28.run(buf264, buf265, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_150], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_490, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 512, 1, 1), (512, 1, 1, 1))
        buf267 = buf266; del buf266  # reuse
        buf268 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_150, input_151, input_152], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf267, primals_491, primals_492, primals_493, primals_494, primals_495, buf268, 2048, grid=grid(2048), stream=stream0)
        del primals_491
        del primals_495
        # Topologically Sorted Source Nodes: [input_153], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_496, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 128, 1, 1), (128, 1, 1, 1))
        buf270 = buf269; del buf269  # reuse
        buf271 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_153, input_154, input_155], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf270, primals_497, primals_498, primals_499, primals_500, primals_501, buf271, 512, grid=grid(512), stream=stream0)
        del primals_497
        del primals_501
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 32, 1, 1), (32, 1, 1, 1))
        buf273 = buf272; del buf272  # reuse
        buf274 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 448)  # alias
        # Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf273, primals_503, primals_504, primals_505, primals_506, primals_507, buf274, 128, grid=grid(128), stream=stream0)
        del primals_503
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf264, primals_508, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 1024, 2, 2), (4096, 4, 2, 1))
        buf276 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf275, primals_509, primals_510, primals_511, primals_512, buf276, 16384, grid=grid(16384), stream=stream0)
        del primals_512
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_513, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf277, (4, 1024, 2, 2), (4096, 4, 2, 1))
        buf278 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_56], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_25.run(buf274, buf278, 4096, grid=grid(4096), stream=stream0)
        buf279 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_162, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_26.run(buf278, buf277, primals_514, primals_515, primals_516, primals_517, buf279, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_518, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 2048, 2, 2), (8192, 4, 2, 1))
        buf281 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_166, x_167, add_14, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_30.run(buf264, buf280, primals_519, primals_520, primals_521, primals_522, buf281, 32768, grid=grid(32768), stream=stream0)
        buf282 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_28.run(buf281, buf282, 8192, grid=grid(8192), stream=stream0)
        # Topologically Sorted Source Nodes: [input_160], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_523, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 512, 1, 1), (512, 1, 1, 1))
        buf284 = buf283; del buf283  # reuse
        buf285 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_160, input_161, input_162], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf284, primals_524, primals_525, primals_526, primals_527, primals_528, buf285, 2048, grid=grid(2048), stream=stream0)
        del primals_524
        del primals_528
        # Topologically Sorted Source Nodes: [input_163], Original ATen: [aten.convolution]
        buf286 = extern_kernels.convolution(buf285, primals_529, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf286, (4, 128, 1, 1), (128, 1, 1, 1))
        buf287 = buf286; del buf286  # reuse
        buf288 = empty_strided_cuda((4, 128, 1, 1), (128, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_163, input_164, input_165], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf287, primals_530, primals_531, primals_532, primals_533, primals_534, buf288, 512, grid=grid(512), stream=stream0)
        del primals_530
        del primals_534
        # Topologically Sorted Source Nodes: [input_166], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, primals_535, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 32, 1, 1), (32, 1, 1, 1))
        buf290 = buf289; del buf289  # reuse
        buf291 = reinterpret_tensor(buf299, (4, 32, 1, 1), (512, 1, 1, 1), 480)  # alias
        # Topologically Sorted Source Nodes: [input_166, input_167, input_168], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_sigmoid_4.run(buf290, primals_536, primals_537, primals_538, primals_539, primals_540, buf291, 128, grid=grid(128), stream=stream0)
        del primals_536
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf281, primals_541, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 1024, 2, 2), (4096, 4, 2, 1))
        buf293 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf292, primals_542, primals_543, primals_544, primals_545, buf293, 16384, grid=grid(16384), stream=stream0)
        del primals_545
        # Topologically Sorted Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_546, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf294, (4, 1024, 2, 2), (4096, 4, 2, 1))
        buf295 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ss_60], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_25.run(buf291, buf295, 4096, grid=grid(4096), stream=stream0)
        buf296 = empty_strided_cuda((4, 1024, 2, 2), (4096, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_173, x_174, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_relu_26.run(buf295, buf294, primals_547, primals_548, primals_549, primals_550, buf296, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, primals_551, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (4, 2048, 2, 2), (8192, 4, 2, 1))
        buf298 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.float32)
        buf305 = empty_strided_cuda((4, 2048, 2, 2), (8192, 4, 2, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_177, x_178, add_15, x_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_threshold_backward_31.run(buf281, buf297, primals_552, primals_553, primals_554, primals_555, buf298, buf305, 32768, grid=grid(32768), stream=stream0)
        buf300 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_28.run(buf298, buf300, 8192, grid=grid(8192), stream=stream0)
        del buf298
        # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_556, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 1024, 1, 1), (1024, 1, 1, 1))
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf299, primals_561, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 1024, 1, 1), (1024, 1, 1, 1))
        buf303 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.float32)
        buf304 = buf303; del buf303  # reuse
        # Topologically Sorted Source Nodes: [input_170, input_171, input_173, input_174, add_16, v], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_mul_relu_32.run(buf304, buf301, primals_557, primals_558, primals_559, primals_560, buf302, primals_562, primals_563, primals_564, primals_565, 4096, grid=grid(4096), stream=stream0)
    return (reinterpret_tensor(buf304, (4, 1024), (1024, 1), 0), primals_1, primals_3, primals_4, primals_5, primals_6, primals_8, primals_10, primals_11, primals_12, primals_14, primals_16, primals_17, primals_18, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_46, primals_48, primals_49, primals_50, primals_52, primals_54, primals_55, primals_56, primals_58, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_85, primals_87, primals_88, primals_89, primals_91, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_114, primals_115, primals_116, primals_118, primals_120, primals_121, primals_122, primals_124, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_150, primals_152, primals_153, primals_154, primals_156, primals_158, primals_159, primals_160, primals_162, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_195, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_218, primals_219, primals_220, primals_222, primals_224, primals_225, primals_226, primals_228, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_251, primals_252, primals_253, primals_255, primals_257, primals_258, primals_259, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_287, primals_289, primals_290, primals_291, primals_293, primals_295, primals_296, primals_297, primals_299, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_326, primals_328, primals_329, primals_330, primals_332, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_355, primals_356, primals_357, primals_359, primals_361, primals_362, primals_363, primals_365, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_388, primals_389, primals_390, primals_392, primals_394, primals_395, primals_396, primals_398, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_421, primals_422, primals_423, primals_425, primals_427, primals_428, primals_429, primals_431, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_454, primals_455, primals_456, primals_458, primals_460, primals_461, primals_462, primals_464, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_490, primals_492, primals_493, primals_494, primals_496, primals_498, primals_499, primals_500, primals_502, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_525, primals_526, primals_527, primals_529, primals_531, primals_532, primals_533, primals_535, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, buf1, buf2, buf3, buf4, buf6, buf8, buf9, buf11, buf12, buf14, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf24, buf26, buf28, buf29, buf31, buf32, buf34, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf44, buf46, buf47, buf49, buf50, buf52, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf62, buf64, buf65, buf67, buf68, buf70, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf80, buf82, buf84, buf85, buf87, buf88, buf90, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf100, buf102, buf103, buf105, buf106, buf108, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf118, buf120, buf121, buf123, buf124, buf126, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf136, buf138, buf139, buf141, buf142, buf144, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf154, buf156, buf158, buf159, buf161, buf162, buf164, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf174, buf176, buf177, buf179, buf180, buf182, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf192, buf194, buf195, buf197, buf198, buf200, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf210, buf212, buf213, buf215, buf216, buf218, buf220, buf221, buf222, buf223, buf224, buf225, buf226, buf228, buf230, buf231, buf233, buf234, buf236, buf238, buf239, buf240, buf241, buf242, buf243, buf244, buf246, buf248, buf249, buf251, buf252, buf254, buf256, buf257, buf258, buf259, buf260, buf261, buf262, buf264, buf265, buf267, buf268, buf270, buf271, buf273, buf275, buf276, buf277, buf278, buf279, buf280, buf281, buf282, buf284, buf285, buf287, buf288, buf290, buf292, buf293, buf294, buf295, buf296, buf297, buf299, buf300, buf301, buf302, buf305, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
