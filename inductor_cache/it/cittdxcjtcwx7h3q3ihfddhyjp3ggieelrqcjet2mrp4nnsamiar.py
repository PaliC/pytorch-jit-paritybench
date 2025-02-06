# AOT ID: ['11_inference']
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


# kernel path: inductor_cache/wt/cwthat4gscdfuuf2edsoqbo46743imc7puru6dhcydy5w4ccfwmn.py
# Topologically Sorted Source Nodes: [keys_1, sum_1, add_1], Original ATen: [aten.sigmoid, aten.sum, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_1
#   keys_1 => sigmoid_1
#   sum_1 => sum_1
# Graph fragment:
#   %sigmoid_1 : [num_users=5] = call_function[target=torch.ops.aten.sigmoid.default](args = (%permute_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sigmoid_1, [2]), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-06), kwargs = {})
triton_poi_fused_add_sigmoid_sum_0 = async_compile.triton('triton_poi_fused_add_sigmoid_sum_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sigmoid_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sigmoid_sum_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp2 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp8 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp1 + tmp3
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp4 + tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp7 + tmp9
    tmp11 = 1e-06
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4ui6jodjgt5ajnbsxq4nrwbird2yfaxo6engdpxnwnb2btioof.py
# Topologically Sorted Source Nodes: [queries_1, matmul_1, add, einsum, add_4, normalizer_row_refine], Original ATen: [aten.sigmoid, aten.clone, aten.add]
# Source node to ATen node mapping:
#   add => add
#   add_4 => add_4
#   einsum => clone
#   matmul_1 => clone_6
#   normalizer_row_refine => clone_2
#   queries_1 => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=5] = call_function[target=torch.ops.aten.sigmoid.default](args = (%permute,), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_2,), kwargs = {memory_format: torch.contiguous_format})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, 1e-06), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add,), kwargs = {memory_format: torch.contiguous_format})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid, 1e-06), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_sigmoid_1 = async_compile.triton('triton_poi_fused_add_clone_sigmoid_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_sigmoid_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_sigmoid_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 16*x1 + 64*x3), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1e-06
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp1, xmask)
    tl.store(out_ptr2 + (x4), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qp/cqprcw4dnls4iaeza2khwbx2j37sbauctof2qwlqi6vn7lujjjd2.py
# Topologically Sorted Source Nodes: [queries_1, mul_1, sum_4, add_7, sum_2, add_3], Original ATen: [aten.sigmoid, aten.mul, aten.sum, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_3
#   add_7 => add_7
#   mul_1 => mul_3
#   queries_1 => sigmoid
#   sum_2 => sum_2
#   sum_4 => sum_4
# Graph fragment:
#   %sigmoid : [num_users=5] = call_function[target=torch.ops.aten.sigmoid.default](args = (%permute,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %unsqueeze_4), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [2]), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_4, 1e-06), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sigmoid, [2]), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_2, 1e-06), kwargs = {})
triton_poi_fused_add_mul_sigmoid_sum_2 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_sum_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_sum_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x3 = (xindex % 16)
    x4 = xindex // 4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + 64*x2), xmask)
    tmp2 = tl.load(in_ptr1 + (4*x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (16 + x3 + 64*x2), xmask)
    tmp10 = tl.load(in_ptr1 + (1 + 4*x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (32 + x3 + 64*x2), xmask)
    tmp17 = tl.load(in_ptr1 + (2 + 4*x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (48 + x3 + 64*x2), xmask)
    tmp24 = tl.load(in_ptr1 + (3 + 4*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp3 / tmp2
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp1 * tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp11 = tmp3 / tmp10
    tmp12 = tmp11 * tmp5
    tmp13 = tmp9 * tmp12
    tmp14 = tmp7 + tmp13
    tmp16 = tl.sigmoid(tmp15)
    tmp18 = tmp3 / tmp17
    tmp19 = tmp18 * tmp5
    tmp20 = tmp16 * tmp19
    tmp21 = tmp14 + tmp20
    tmp23 = tl.sigmoid(tmp22)
    tmp25 = tmp3 / tmp24
    tmp26 = tmp25 * tmp5
    tmp27 = tmp23 * tmp26
    tmp28 = tmp21 + tmp27
    tmp29 = 1e-06
    tmp30 = tmp28 + tmp29
    tmp31 = tmp1 + tmp9
    tmp32 = tmp31 + tmp16
    tmp33 = tmp32 + tmp23
    tmp34 = tmp33 + tmp29
    tl.store(out_ptr0 + (x5), tmp30, xmask)
    tl.store(out_ptr1 + (x5), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kp/ckp3ovtrr3xecncsvs2be3bncaqfwlljdvv6hncflbeumwgv2exu.py
# Topologically Sorted Source Nodes: [keys_1, add_6, normalizer_col_refine, add_2, einsum_1], Original ATen: [aten.sigmoid, aten.add, aten.clone]
# Source node to ATen node mapping:
#   add_2 => add_2
#   add_6 => add_6
#   einsum_1 => clone_1
#   keys_1 => sigmoid_1
#   normalizer_col_refine => clone_3
# Graph fragment:
#   %sigmoid_1 : [num_users=5] = call_function[target=torch.ops.aten.sigmoid.default](args = (%permute_1,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid_1, 1e-06), kwargs = {})
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_6,), kwargs = {memory_format: torch.contiguous_format})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sigmoid_1, 1e-06), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_add_clone_sigmoid_3 = async_compile.triton('triton_poi_fused_add_clone_sigmoid_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_sigmoid_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clone_sigmoid_3(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 16*x1 + 64*x3), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = 1e-06
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr0 + (x4), tmp3, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lw/clwujgjma4btrhu7c7w7kkqyk2iskqshebzhjbc364dhhpszcwyx.py
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_15, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_15, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_4 = async_compile.triton('triton_poi_fused__softmax_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/io/cioisgq5k3b3nqo4734vstnsxtbfommh5ppwybhzzp7ngimd63ba.py
# Topologically Sorted Source Nodes: [softmax, normalizer_col_refine_1], Original ATen: [aten._softmax, aten.mul]
# Source node to ATen node mapping:
#   normalizer_col_refine_1 => mul_5
#   softmax => div, sum_5
# Graph fragment:
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 4), kwargs = {})
triton_poi_fused__softmax_mul_5 = async_compile.triton('triton_poi_fused__softmax_mul_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_mul_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_mul_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tmp9 = 4.0
    tmp10 = tmp8 * tmp9
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/42/c424qfsgvped36pmiudcr45t2mujz53t75i2bbm75sd4ejv6bxqd.py
# Topologically Sorted Source Nodes: [keys_1, mul, sum_3, add_5], Original ATen: [aten.sigmoid, aten.mul, aten.sum, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_5
#   keys_1 => sigmoid_1
#   mul => mul_2
#   sum_3 => sum_3
# Graph fragment:
#   %sigmoid_1 : [num_users=5] = call_function[target=torch.ops.aten.sigmoid.default](args = (%permute_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid_1, %unsqueeze_2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [2]), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_3, 1e-06), kwargs = {})
triton_poi_fused_add_mul_sigmoid_sum_6 = async_compile.triton('triton_poi_fused_add_mul_sigmoid_sum_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_sum_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sigmoid_sum_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 16
    x3 = (xindex % 16)
    x4 = xindex // 4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + 64*x2), xmask)
    tmp2 = tl.load(in_ptr1 + (4*x4), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (16 + x3 + 64*x2), xmask)
    tmp10 = tl.load(in_ptr1 + (1 + 4*x4), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (32 + x3 + 64*x2), xmask)
    tmp17 = tl.load(in_ptr1 + (2 + 4*x4), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (48 + x3 + 64*x2), xmask)
    tmp24 = tl.load(in_ptr1 + (3 + 4*x4), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = tmp3 / tmp2
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tmp7 = tmp1 * tmp6
    tmp9 = tl.sigmoid(tmp8)
    tmp11 = tmp3 / tmp10
    tmp12 = tmp11 * tmp5
    tmp13 = tmp9 * tmp12
    tmp14 = tmp7 + tmp13
    tmp16 = tl.sigmoid(tmp15)
    tmp18 = tmp3 / tmp17
    tmp19 = tmp18 * tmp5
    tmp20 = tmp16 * tmp19
    tmp21 = tmp14 + tmp20
    tmp23 = tl.sigmoid(tmp22)
    tmp25 = tmp3 / tmp24
    tmp26 = tmp25 * tmp5
    tmp27 = tmp23 * tmp26
    tmp28 = tmp21 + tmp27
    tmp29 = 1e-06
    tmp30 = tmp28 + tmp29
    tl.store(out_ptr0 + (x5), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rx/crxvo4wjanzesh6wjflmv56q2totg35dqrozeylydtzvhicbcrew.py
# Topologically Sorted Source Nodes: [kv], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   kv => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + (x2 + 4*y3), tmp1, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/yu/cyulbt3wj3frlro7ng65lq4vvpkwu2xioa3hbs7m7van62lnsknj.py
# Topologically Sorted Source Nodes: [mul_4, kv], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   kv => clone_5
#   mul_4 => mul_6
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, %unsqueeze_6), kwargs = {})
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_8 = async_compile.triton('triton_poi_fused_clone_mul_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_8(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex // 4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 16*x1 + 64*x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x5), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/54/c54mq3jse3i57ffjc5jpcgprae3pdjyzjq7b7nvbymhex6nkacz3.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x => clone_7
# Graph fragment:
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_24,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_9 = async_compile.triton('triton_poi_fused_clone_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x5 = xindex // 4
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x5), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x5), xmask, eviction_policy='evict_last')
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = tmp2 / tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 * tmp5
    tmp8 = tmp7 * tmp4
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp6 * tmp9
    tl.store(out_ptr0 + (x0 + 4*x2 + 16*x1 + 64*x3), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [keys_1, sum_1, add_1], Original ATen: [aten.sigmoid, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sigmoid_sum_0.run(arg1_1, buf1, 64, grid=grid(64), stream=stream0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf11 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf16 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [queries_1, matmul_1, add, einsum, add_4, normalizer_row_refine], Original ATen: [aten.sigmoid, aten.clone, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_sigmoid_1.run(arg0_1, buf0, buf11, buf16, 256, grid=grid(256), stream=stream0)
        buf2 = empty_strided_cuda((16, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf1, (16, 4, 1), (4, 1, 0), 0), out=buf2)
        buf4 = buf1; del buf1  # reuse
        buf14 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [queries_1, mul_1, sum_4, add_7, sum_2, add_3], Original ATen: [aten.sigmoid, aten.mul, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sum_2.run(arg0_1, buf2, buf4, buf14, 64, grid=grid(64), stream=stream0)
        del arg0_1
        buf3 = buf0; del buf0  # reuse
        buf13 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [keys_1, add_6, normalizer_col_refine, add_2, einsum_1], Original ATen: [aten.sigmoid, aten.add, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_sigmoid_3.run(arg1_1, buf3, buf13, 256, grid=grid(256), stream=stream0)
        buf5 = empty_strided_cuda((16, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [normalizer_col_refine], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf4, (16, 4, 1), (4, 1, 0), 0), out=buf5)
        buf6 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf5, buf6, 64, grid=grid(64), stream=stream0)
        buf7 = reinterpret_tensor(buf5, (4, 4, 4), (16, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [softmax, normalizer_col_refine_1], Original ATen: [aten._softmax, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_mul_5.run(buf6, buf7, 64, grid=grid(64), stream=stream0)
        buf15 = reinterpret_tensor(buf6, (16, 4, 1), (4, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf13, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf14, (16, 4, 1), (4, 1, 0), 0), out=buf15)
        buf17 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [keys_1, mul, sum_3, add_5], Original ATen: [aten.sigmoid, aten.mul, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sigmoid_sum_6.run(arg1_1, buf15, buf17, 64, grid=grid(64), stream=stream0)
        del buf15
        buf8 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [kv], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(arg1_1, buf8, 64, 4, grid=grid(64, 4), stream=stream0)
        del arg1_1
        buf9 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [mul_4, kv], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_8.run(arg2_1, buf7, buf9, 256, grid=grid(256), stream=stream0)
        del arg2_1
        buf18 = reinterpret_tensor(buf7, (16, 4, 1), (4, 1, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [normalizer_row_refine], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf16, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf17, (16, 4, 1), (4, 1, 0), 0), out=buf18)
        del buf17
        buf10 = reinterpret_tensor(buf16, (16, 4, 4), (16, 4, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [kv], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf9, (16, 4, 4), (16, 4, 1), 0), out=buf10)
        del buf8
        buf12 = reinterpret_tensor(buf9, (16, 4, 4), (16, 4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (16, 4, 4), (16, 4, 1), 0), buf10, out=buf12)
        del buf10
        buf19 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_9.run(buf12, buf2, buf18, buf19, 256, grid=grid(256), stream=stream0)
        del buf12
        del buf18
        del buf2
    return (buf19, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
