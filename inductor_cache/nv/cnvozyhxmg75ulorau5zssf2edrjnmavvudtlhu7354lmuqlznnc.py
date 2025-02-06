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


# kernel path: inductor_cache/uw/cuwh6hcvj5nac35mr3yr53cozu5of4o3ljharncl6kqy5ob3d56m.py
# Topologically Sorted Source Nodes: [weights], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   weights => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%primals_1, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_0 = async_compile.triton('triton_poi_fused__softmax_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (2))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (3))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp8 = triton_helpers.maximum(tmp5, tmp7)
    tmp11 = triton_helpers.maximum(tmp8, tmp10)
    tmp12 = tmp0 - tmp11
    tmp13 = libdevice.exp(tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tj/ctjnu4r2c6ir37kfovuftntlyt5a2kmpaayarlp4nqn5e6dr4id4.py
# Topologically Sorted Source Nodes: [weights], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   weights => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_1 = async_compile.triton('triton_poi_fused__softmax_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (2))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (3))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp5 = tmp2 + tmp4
    tmp8 = tmp5 + tmp7
    tmp11 = tmp8 + tmp10
    tmp12 = tmp0 / tmp11
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wh/cwhh3pkcjvbrrx37xfagevkorcunpnwbc7xshtftet4ya4ue26rq.py
# Topologically Sorted Source Nodes: [mode_1h], Original ATen: [aten.arange, aten.eq]
# Source node to ATen node mapping:
#   mode_1h => eq, iota
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %eq : [num_users=2] = call_function[target=torch.ops.aten.eq.Tensor](args = (%unsqueeze, %iota), kwargs = {})
triton_poi_fused_arange_eq_2 = async_compile.triton('triton_poi_fused_arange_eq_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_eq_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_eq_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fp/cfp2qa45gjgdhnhdxkfe5ad343rbapjmk4qtmyx4y7xotobc4yfn.py
# Topologically Sorted Source Nodes: [exp, mul, scale_sample, mul_1, loc_sample, mul_2, z], Original ATen: [aten.exp, aten.mul, aten.sum, aten.add]
# Source node to ATen node mapping:
#   exp => exp_1
#   loc_sample => sum_3
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   scale_sample => sum_2
#   z => add
# Graph fragment:
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%primals_3,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %unsqueeze_1), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, %unsqueeze_1), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [1]), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%randn, %sum_2), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %sum_3), kwargs = {})
triton_poi_fused_add_exp_mul_sum_3 = async_compile.triton('triton_poi_fused_add_exp_mul_sum_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*i1', 'in_ptr3': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_sum_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_mul_sum_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp3 = tl.load(in_ptr2 + (0)).to(tl.int1)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp8 = tl.load(in_ptr1 + (4 + x0), xmask)
    tmp10 = tl.load(in_ptr2 + (1)).to(tl.int1)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + (8 + x0), xmask)
    tmp18 = tl.load(in_ptr2 + (2)).to(tl.int1)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (12 + x0), xmask)
    tmp26 = tl.load(in_ptr2 + (3)).to(tl.int1)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp33 = tl.load(in_ptr3 + (x0), xmask)
    tmp35 = tl.load(in_ptr3 + (4 + x0), xmask)
    tmp38 = tl.load(in_ptr3 + (8 + x0), xmask)
    tmp41 = tl.load(in_ptr3 + (12 + x0), xmask)
    tmp2 = libdevice.exp(tmp1)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tmp5.to(tl.float64)
    tmp7 = tmp2 * tmp6
    tmp9 = libdevice.exp(tmp8)
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp12.to(tl.float64)
    tmp14 = tmp9 * tmp13
    tmp15 = tmp7 + tmp14
    tmp17 = libdevice.exp(tmp16)
    tmp20 = tmp19.to(tl.int64)
    tmp21 = tmp20.to(tl.float64)
    tmp22 = tmp17 * tmp21
    tmp23 = tmp15 + tmp22
    tmp25 = libdevice.exp(tmp24)
    tmp28 = tmp27.to(tl.int64)
    tmp29 = tmp28.to(tl.float64)
    tmp30 = tmp25 * tmp29
    tmp31 = tmp23 + tmp30
    tmp32 = tmp0 * tmp31
    tmp34 = tmp33 * tmp6
    tmp36 = tmp35 * tmp13
    tmp37 = tmp34 + tmp36
    tmp39 = tmp38 * tmp21
    tmp40 = tmp37 + tmp39
    tmp42 = tmp41 * tmp29
    tmp43 = tmp40 + tmp42
    tmp44 = tmp32 + tmp43
    tl.store(in_out_ptr0 + (x0), tmp44, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c3/cc3yyvlgkktq7qldehpkprvyvsoyh7aayv7sw2yzfmay5cjlp5lz.py
# Topologically Sorted Source Nodes: [exp, sub, eps, wrapped_mul, log, add_1, pow_1, sum_3, mul_3, sub_1, sum_4, log_p], Original ATen: [aten.exp, aten.sub, aten.div, aten.mul, aten.log, aten.add, aten.pow, aten.sum]
# Source node to ATen node mapping:
#   add_1 => add_1
#   eps => div_1
#   exp => exp_1
#   log => log_1
#   log_p => sub_3
#   mul_3 => mul_4
#   pow_1 => pow_1
#   sub => sub_1
#   sub_1 => sub_2
#   sum_3 => sum_4
#   sum_4 => sum_5
#   wrapped_mul => full_default
# Graph fragment:
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%primals_3,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %primals_2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %exp_1), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -3.6757541328186907), kwargs = {dtype: torch.float64, layout: torch.strided, device: cpu, pin_memory: False})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%div,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%full_default, %log_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%div_1, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [2]), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_4, 0.5), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %mul_4), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%primals_3, [2]), kwargs = {})
#   %sub_3 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_2, %sum_5), kwargs = {})
triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_4 = async_compile.triton('triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'in_ptr1': '*fp64', 'in_ptr2': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp10 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (3))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp28 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp3 = tmp1 - tmp2
    tmp5 = libdevice.exp(tmp4)
    tmp6 = tmp3 / tmp5
    tmp7 = tmp6 * tmp6
    tmp11 = tmp9 - tmp10
    tmp13 = libdevice.exp(tmp12)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tmp7 + tmp15
    tmp20 = tmp18 - tmp19
    tmp22 = libdevice.exp(tmp21)
    tmp23 = tmp20 / tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tmp16 + tmp24
    tmp29 = tmp27 - tmp28
    tmp31 = libdevice.exp(tmp30)
    tmp32 = tmp29 / tmp31
    tmp33 = tmp32 * tmp32
    tmp34 = tmp25 + tmp33
    tmp36 = libdevice.log(tmp35)
    tmp37 = tl.full([1], -3.6757541328186907, tl.float64)
    tmp38 = tmp37 + tmp36
    tmp39 = tl.full([1], 0.5, tl.float64)
    tmp40 = tmp34 * tmp39
    tmp41 = tmp38 - tmp40
    tmp42 = tmp4 + tmp12
    tmp43 = tmp42 + tmp21
    tmp44 = tmp43 + tmp30
    tmp45 = tmp41 - tmp44
    tl.store(in_out_ptr0 + (x0), tmp45, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y4/cy4r77bepx4czusl3tehhb6il3xr36xqndgdoeghckx2ksnxe4lw.py
# Topologically Sorted Source Nodes: [log_p_1], Original ATen: [aten.logsumexp]
# Source node to ATen node mapping:
#   log_p_1 => abs_1, add_2, amax_1, eq_1, exp_3, full_default_1, log_2, sub_4, sum_6, where
# Graph fragment:
#   %amax_1 : [num_users=2] = call_function[target=torch.ops.aten.amax.default](args = (%sub_3, [1], True), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%amax_1,), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%abs_1, inf), kwargs = {})
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq_1, %full_default_1, %amax_1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %where), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_3, [1]), kwargs = {})
#   %log_2 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_6,), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_2, %squeeze), kwargs = {})
triton_poi_fused_logsumexp_5 = async_compile.triton('triton_poi_fused_logsumexp_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp64', 'out_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_logsumexp_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_logsumexp_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (1))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (2))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (3))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp11 = tl_math.abs(tmp10)
    tmp12 = tl.full([1], float("inf"), tl.float64)
    tmp13 = tmp11 == tmp12
    tmp14 = tl.full([1], 0.0, tl.float64)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = tmp1 - tmp15
    tmp17 = libdevice.exp(tmp16)
    tmp18 = tmp3 - tmp15
    tmp19 = libdevice.exp(tmp18)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp6 - tmp15
    tmp22 = libdevice.exp(tmp21)
    tmp23 = tmp20 + tmp22
    tmp24 = tmp9 - tmp15
    tmp25 = libdevice.exp(tmp24)
    tmp26 = tmp23 + tmp25
    tmp27 = libdevice.log(tmp26)
    tmp28 = tmp27 + tmp15
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp28, None)
''', device_str='cuda')


# kernel path: inductor_cache/4r/c4r3h44nbx2d6rhztvqnjb7l76us5s6tlffoy32vtxemznvfa3ve.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp]
# Source node to ATen node mapping:
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %unsqueeze_4), kwargs = {})
#   %exp_4 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_5,), kwargs = {})
triton_poi_fused_exp_sub_6 = async_compile.triton('triton_poi_fused_exp_sub_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp64', 'in_ptr0': '*fp64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_sub_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_sub_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp4 = libdevice.exp(tmp3)
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 4), (4, 1))
    assert_size_stride(primals_2, (1, 4, 4), (16, 4, 1))
    assert_size_stride(primals_3, (1, 4, 4), (16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 4), (4, 1), torch.float64)
        # Topologically Sorted Source Nodes: [weights], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(primals_1, buf0, 4, grid=grid(4), stream=stream0)
        buf1 = empty_strided_cuda((1, 4), (4, 1), torch.float64)
        # Topologically Sorted Source Nodes: [weights], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf0, buf1, 4, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [mode], Original ATen: [aten.multinomial]
        buf2 = torch.ops.aten.multinomial.default(reinterpret_tensor(buf1, (4, ), (1, ), 0), 1, True)
        buf3 = buf2
        del buf2
        buf4 = empty_strided_cuda((1, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [mode_1h], Original ATen: [aten.arange, aten.eq]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_eq_2.run(buf3, buf4, 4, grid=grid(4), stream=stream0)
        del buf3
        # Topologically Sorted Source Nodes: [eps_], Original ATen: [aten.randn]
        buf5 = torch.ops.aten.randn.default([1, 4], dtype=torch.float64, device=device(type='cuda', index=0), pin_memory=False)
        buf6 = buf5
        del buf5
        buf7 = buf0; del buf0  # reuse
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [exp, mul, scale_sample, mul_1, loc_sample, mul_2, z], Original ATen: [aten.exp, aten.mul, aten.sum, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_mul_sum_3.run(buf8, buf6, primals_3, buf4, primals_2, 4, grid=grid(4), stream=stream0)
        buf10 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [exp, sub, eps, wrapped_mul, log, add_1, pow_1, sum_3, mul_3, sub_1, sum_4, log_p], Original ATen: [aten.exp, aten.sub, aten.div, aten.mul, aten.log, aten.add, aten.pow, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_4.run(buf10, buf8, primals_2, primals_3, 4, grid=grid(4), stream=stream0)
        buf11 = empty_strided_cuda((1, ), (1, ), torch.float64)
        # Topologically Sorted Source Nodes: [log_p_1], Original ATen: [aten.logsumexp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_logsumexp_5.run(buf10, buf11, 1, grid=grid(1), stream=stream0)
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sub, aten.exp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_sub_6.run(buf12, buf11, 4, grid=grid(4), stream=stream0)
    return (buf8, buf11, primals_1, primals_2, primals_3, buf4, buf6, buf8, buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float64)
    primals_2 = rand_strided((1, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float64)
    primals_3 = rand_strided((1, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float64)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
