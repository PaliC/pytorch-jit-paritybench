# AOT ID: ['6_inference']
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


# kernel path: inductor_cache/yb/cyby23gk37lh3q2omkg3syky2idinvidotxdq2jcdxdofod2p23w.py
# Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   p_s => exp
# Graph fragment:
#   %mul_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 1), kwargs = {})
#   %amax_default_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_1, [1], True), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_1, %amax_default_1), kwargs = {})
#   %div_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_tensor_1, 2), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_tensor_1,), kwargs = {})
triton_poi_fused__softmax_0 = async_compile.triton('triton_poi_fused__softmax_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
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
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = tl_math.exp(tmp16)
    tl.store(out_ptr0 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wx/cwxcrbnklwz6w53ralqr4hsjdu4fcj5udnv533dnsk72zb62kmtp.py
# Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   p_s => div_1, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_1 = async_compile.triton('triton_poi_fused__softmax_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m2/cm2vwwblc5k43xikfawrr6sh5jw4x2aemqoiek5nfbqpccfblgz2.py
# Topologically Sorted Source Nodes: [neg, truediv_2, K, sum_1, x], Original ATen: [aten.neg, aten.div, aten.exp, aten.sum]
# Source node to ATen node mapping:
#   K => exp_2
#   neg => neg
#   sum_1 => sum_3
#   truediv_2 => div_4
#   x => div_5
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%_cdist_forward,), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg, 0.1), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_4,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [1], True), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_2, %sum_3), kwargs = {})
triton_poi_fused_div_exp_neg_sum_2 = async_compile.triton('triton_poi_fused_div_exp_neg_sum_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_exp_neg_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_exp_neg_sum_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp5 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp2 = 10.0
    tmp3 = tmp1 * tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp6 = -tmp5
    tmp7 = tmp6 * tmp2
    tmp8 = tl_math.exp(tmp7)
    tmp10 = -tmp9
    tmp11 = tmp10 * tmp2
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tmp8 + tmp12
    tmp15 = -tmp14
    tmp16 = tmp15 * tmp2
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp13 + tmp17
    tmp20 = -tmp19
    tmp21 = tmp20 * tmp2
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp18 + tmp22
    tmp24 = tmp4 / tmp23
    tl.store(out_ptr0 + (x3), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tr/ctrvzt4dicqploekbpgeeu27fhd3tbvf5u5mruv22ijsin2ascle.py
# Topologically Sorted Source Nodes: [sum_2, x_1], Original ATen: [aten.sum, aten.div]
# Source node to ATen node mapping:
#   sum_2 => sum_4
#   x_1 => div_6
# Graph fragment:
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_5, [0], True), kwargs = {})
#   %div_6 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_5, %sum_4), kwargs = {})
triton_poi_fused_div_sum_3 = async_compile.triton('triton_poi_fused_div_sum_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sum_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (128 + x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (192 + x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xw/cxwyp66uo5pu3jr3ejiofdhzy5a2vtzjh24knemgqtjmob22i36k.py
# Topologically Sorted Source Nodes: [sum_40, x_39, mul, sum_41, emd_loss], Original ATen: [aten.sum, aten.div, aten.mul]
# Source node to ATen node mapping:
#   emd_loss => mul_1
#   mul => mul
#   sum_40 => sum_42
#   sum_41 => sum_43
#   x_39 => div_44
# Graph fragment:
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_43, [0], True), kwargs = {})
#   %div_44 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_43, %sum_42), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_44, %_cdist_forward), kwargs = {})
#   %sum_43 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_43, 0.001), kwargs = {})
triton_per_fused_div_mul_sum_4 = async_compile.triton('triton_per_fused_div_mul_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_mul_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_mul_sum_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 1
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
    r0 = (rindex % 64)
    tmp0 = tl.load(in_ptr0 + (r2), None)
    tmp1 = tl.load(in_ptr0 + (r0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (64 + r0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (128 + r0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (192 + r0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (r2), None)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tmp10 = tmp8 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 0.001
    tmp15 = tmp13 * tmp14
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([1], 0, tl.int32)), tmp15, None)
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
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_t], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg1_1, buf1, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf0, buf2, 256, grid=grid(256), stream=stream0)
        buf3 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [p_t], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf1, buf3, 256, grid=grid(256), stream=stream0)
        del buf1
        # Topologically Sorted Source Nodes: [p_s, p_t, Wxy], Original ATen: [aten._softmax, aten._cdist_forward]
        buf4 = torch.ops.aten._cdist_forward.default(buf2, buf3, 1.0, None)
        buf5 = buf4
        del buf4
        buf6 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [neg, truediv_2, K, sum_1, x], Original ATen: [aten.neg, aten.div, aten.exp, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_exp_neg_sum_2.run(buf5, buf6, 256, grid=grid(256), stream=stream0)
        buf7 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [sum_2, x_1], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf6, buf7, 256, grid=grid(256), stream=stream0)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [sum_3, x_2], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf7, buf8, 256, grid=grid(256), stream=stream0)
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sum_4, x_3], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf8, buf9, 256, grid=grid(256), stream=stream0)
        buf10 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [sum_5, x_4], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf9, buf10, 256, grid=grid(256), stream=stream0)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sum_6, x_5], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf10, buf11, 256, grid=grid(256), stream=stream0)
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [sum_7, x_6], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf11, buf12, 256, grid=grid(256), stream=stream0)
        buf13 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [sum_8, x_7], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf12, buf13, 256, grid=grid(256), stream=stream0)
        buf14 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [sum_9, x_8], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf13, buf14, 256, grid=grid(256), stream=stream0)
        buf15 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [sum_10, x_9], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf14, buf15, 256, grid=grid(256), stream=stream0)
        buf16 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [sum_11, x_10], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf15, buf16, 256, grid=grid(256), stream=stream0)
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [sum_12, x_11], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf16, buf17, 256, grid=grid(256), stream=stream0)
        buf18 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [sum_13, x_12], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf17, buf18, 256, grid=grid(256), stream=stream0)
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [sum_14, x_13], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf18, buf19, 256, grid=grid(256), stream=stream0)
        buf20 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [sum_15, x_14], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf19, buf20, 256, grid=grid(256), stream=stream0)
        buf21 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [sum_16, x_15], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf20, buf21, 256, grid=grid(256), stream=stream0)
        buf22 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [sum_17, x_16], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf21, buf22, 256, grid=grid(256), stream=stream0)
        buf23 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [sum_18, x_17], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf22, buf23, 256, grid=grid(256), stream=stream0)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [sum_19, x_18], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf23, buf24, 256, grid=grid(256), stream=stream0)
        buf25 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [sum_20, x_19], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf24, buf25, 256, grid=grid(256), stream=stream0)
        buf26 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [sum_21, x_20], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf25, buf26, 256, grid=grid(256), stream=stream0)
        buf27 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [sum_22, x_21], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf26, buf27, 256, grid=grid(256), stream=stream0)
        buf28 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [sum_23, x_22], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf27, buf28, 256, grid=grid(256), stream=stream0)
        buf29 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [sum_24, x_23], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf28, buf29, 256, grid=grid(256), stream=stream0)
        buf30 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [sum_25, x_24], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf29, buf30, 256, grid=grid(256), stream=stream0)
        buf31 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [sum_26, x_25], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf30, buf31, 256, grid=grid(256), stream=stream0)
        buf32 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [sum_27, x_26], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf31, buf32, 256, grid=grid(256), stream=stream0)
        buf33 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [sum_28, x_27], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf32, buf33, 256, grid=grid(256), stream=stream0)
        buf34 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [sum_29, x_28], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf33, buf34, 256, grid=grid(256), stream=stream0)
        buf35 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [sum_30, x_29], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf34, buf35, 256, grid=grid(256), stream=stream0)
        buf36 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [sum_31, x_30], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf35, buf36, 256, grid=grid(256), stream=stream0)
        buf37 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [sum_32, x_31], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf36, buf37, 256, grid=grid(256), stream=stream0)
        buf38 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [sum_33, x_32], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf37, buf38, 256, grid=grid(256), stream=stream0)
        buf39 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [sum_34, x_33], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf38, buf39, 256, grid=grid(256), stream=stream0)
        buf40 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [sum_35, x_34], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf39, buf40, 256, grid=grid(256), stream=stream0)
        buf41 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [sum_36, x_35], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf40, buf41, 256, grid=grid(256), stream=stream0)
        buf42 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [sum_37, x_36], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf41, buf42, 256, grid=grid(256), stream=stream0)
        buf43 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [sum_38, x_37], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf42, buf43, 256, grid=grid(256), stream=stream0)
        buf44 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [sum_39, x_38], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf43, buf44, 256, grid=grid(256), stream=stream0)
        del buf43
        buf45 = empty_strided_cuda((), (), torch.float32)
        buf46 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [sum_40, x_39, mul, sum_41, emd_loss], Original ATen: [aten.sum, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_mul_sum_4.run(buf46, buf44, buf5, 1, 256, grid=grid(1), stream=stream0)
        del buf44
        del buf5
    return (buf46, )


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
