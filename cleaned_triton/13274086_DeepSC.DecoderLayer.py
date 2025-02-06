# AOT ID: ['1_forward']
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

empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: inductor_cache/yo/cyofsbkgqaevvhdvmxyhmaxgaiicq2myopeddsnpwfiywj2wdub5.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 4*y3), tmp2, xmask & ymask)




# kernel path: inductor_cache/2n/c2nsy72zwdgl2i4kp73u62li67wegzwgmlksjxgdqbjydemw4ky3.py
# Topologically Sorted Source Nodes: [scores, mul, scores_1, p_attn], Original ATen: [aten.div, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   mul => mul
#   p_attn => amax, exp, sub, sum_1
#   scores => div
#   scores_1 => add
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_11, 1.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, -1000000000.0), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %mul), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_add_div_mul_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (4*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (1 + 4*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (1 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (2 + 4*x3), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (2 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (3 + 4*x3), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (3 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = -1000000000.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp7 * tmp1
    tmp10 = tmp9 * tmp4
    tmp11 = tmp8 + tmp10
    tmp12 = triton_helpers.maximum(tmp6, tmp11)
    tmp14 = tmp13 * tmp1
    tmp16 = tmp15 * tmp4
    tmp17 = tmp14 + tmp16
    tmp18 = triton_helpers.maximum(tmp12, tmp17)
    tmp20 = tmp19 * tmp1
    tmp22 = tmp21 * tmp4
    tmp23 = tmp20 + tmp22
    tmp24 = triton_helpers.maximum(tmp18, tmp23)
    tmp25 = tmp6 - tmp24
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp11 - tmp24
    tmp28 = tl_math.exp(tmp27)
    tmp29 = tmp26 + tmp28
    tmp30 = tmp17 - tmp24
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp29 + tmp31
    tmp33 = tmp23 - tmp24
    tmp34 = tl_math.exp(tmp33)
    tmp35 = tmp32 + tmp34
    tl.store(out_ptr0 + (x3), tmp24, xmask)
    tl.store(out_ptr1 + (x3), tmp35, xmask)




# kernel path: inductor_cache/ho/chorbhaipqyo63wgxhcqmpiib2l6tbuie63rfncuoxunzpjvfnew.py
# Topologically Sorted Source Nodes: [scores, mul, scores_1, p_attn], Original ATen: [aten.div, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   mul => mul
#   p_attn => amax, div_1, exp, sub
#   scores => div
#   scores_1 => add
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_11, 1.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, -1000000000.0), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%div, %mul), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused__softmax_add_div_mul_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x3 = xindex // 64
    x5 = (xindex % 16)
    x6 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp3 = tl.load(in_ptr0 + (x5 + 16*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = -1000000000.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tmp11 = tmp9 / tmp10
    tl.store(in_out_ptr0 + (x4), tmp11, xmask)




# kernel path: inductor_cache/u6/cu6ttg4y2m4axfsjlc2cnregftdtgn5tvxbu4c2xvyzqfrqsrmzo.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)




# kernel path: inductor_cache/ov/covovhsne62ecy5icqk5az2puc3bkvlnvijmzhik3o5rwffqt56z.py
# Topologically Sorted Source Nodes: [add, x_4], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_1
#   x_4 => var_mean
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %view_17), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_native_layer_norm_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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




# kernel path: inductor_cache/na/cnat6zpjsimj2tsshix5ntineh5yl2h76xxzmqgrwscrmqzeuiot.py
# Topologically Sorted Source Nodes: [add, x_4], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add_1
#   x_4 => add_2, add_3, mul_1, mul_2, rsqrt, sub_1
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %view_17), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_11), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_12), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp6 = 1e-06
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)




# kernel path: inductor_cache/2b/c2bdendzwvcrrcb4fcxou4puoqyoudh3tn7wux6zvml4onylog56.py
# Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_1 => add_5
# Graph fragment:
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_35), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_add_6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)




# kernel path: inductor_cache/ct/cctopwat5bzzrbw7re7ero3gycndqr4n6gkb342ybbsrys6asihd.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_9 => add_6, rsqrt_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_layer_norm_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)




# kernel path: inductor_cache/h5/ch5ijyt5z4xhmr3xg3shv5cjjj3xc4vwty537xw2fg4cwblbp4hp.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_9 => add_6, add_7, mul_4, mul_5, rsqrt_1, sub_3, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_5, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_6,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_1), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %primals_23), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %primals_24), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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




# kernel path: inductor_cache/lw/clwgacjyffqrgywjetyaep7pqqvxgcof5yr7ucn47ln5s2bkqto4.py
# Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x_11 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_37,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def triton_poi_fused_relu_threshold_backward_9(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)







def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, 4), (4, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, 4), (4, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_14, (4, 4), (4, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, 4), (4, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_19, (4, 4), (4, 1))
    assert_size_stride(primals_20, (4, ), (1, ))
    assert_size_stride(primals_21, (4, 4), (4, 1))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    assert_size_stride(primals_25, (4, 4), (4, 1))
    assert_size_stride(primals_26, (4, ), (1, ))
    assert_size_stride(primals_27, (4, 4), (4, 1))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, ), (1, ))
    assert_size_stride(primals_30, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_2, (16, 4), (4, 1), 0), reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf0)
        del primals_3
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_2, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 4), (1, 4), 0), out=buf1)
        del primals_5
        buf2 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_2, (16, 4), (4, 1), 0), reinterpret_tensor(primals_7, (4, 4), (1, 4), 0), out=buf2)
        del primals_7
        buf3 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(16, 4)](buf0, primals_4, buf3, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        del primals_4
        buf4 = reinterpret_tensor(buf0, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(16, 4)](buf1, primals_6, buf4, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        del primals_6
        buf5 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf4, (16, 1, 4), (4, 0, 1), 0), out=buf5)
        buf6 = reinterpret_tensor(buf1, (4, 4, 4, 1), (16, 4, 1, 64), 0); del buf1  # reuse
        buf7 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scores, mul, scores_1, p_attn], Original ATen: [aten.div, aten.mul, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_div_mul_1[grid(64)](buf5, primals_1, buf6, buf7, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf8 = reinterpret_tensor(buf5, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [scores, mul, scores_1, p_attn], Original ATen: [aten.div, aten.mul, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_div_mul_2[grid(256)](buf8, primals_1, buf6, buf7, 256, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_1
        buf9 = reinterpret_tensor(buf7, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(16, 4)](buf2, primals_8, buf9, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        del primals_8
        buf10 = reinterpret_tensor(buf2, (16, 4, 1), (4, 1, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf9, (16, 4, 1), (4, 1, 0), 0), out=buf10)
        buf11 = reinterpret_tensor(buf6, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3[grid(16, 4)](buf10, buf11, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        buf12 = reinterpret_tensor(buf10, (16, 4), (4, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf11, (16, 4), (4, 1), 0), reinterpret_tensor(primals_9, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf12)
        del primals_10
        buf13 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf14 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [add, x_4], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_4[grid(16)](primals_2, buf12, buf13, buf14, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf15 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add, x_4], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_5[grid(64)](primals_2, buf12, buf13, buf14, primals_11, primals_12, buf15, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_12
        buf16 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf15, (16, 4), (4, 1), 0), reinterpret_tensor(primals_14, (4, 4), (1, 4), 0), out=buf16)
        buf17 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_18, (16, 4), (4, 1), 0), reinterpret_tensor(primals_16, (4, 4), (1, 4), 0), out=buf17)
        del primals_16
        buf18 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_18, (16, 4), (4, 1), 0), reinterpret_tensor(primals_19, (4, 4), (1, 4), 0), out=buf18)
        del primals_19
        buf19 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(16, 4)](buf16, primals_15, buf19, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        del primals_15
        buf20 = reinterpret_tensor(buf16, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(16, 4)](buf17, primals_17, buf20, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        del primals_17
        buf21 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf19, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf20, (16, 1, 4), (4, 0, 1), 0), out=buf21)
        buf22 = reinterpret_tensor(buf17, (4, 4, 4, 1), (16, 4, 1, 64), 0); del buf17  # reuse
        buf23 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scores_2, mul_1, scores_3, p_attn_1], Original ATen: [aten.div, aten.mul, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_div_mul_1[grid(64)](buf21, primals_13, buf22, buf23, 64, XBLOCK=64, num_warps=1, num_stages=1)
        buf24 = reinterpret_tensor(buf21, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [scores_2, mul_1, scores_3, p_attn_1], Original ATen: [aten.div, aten.mul, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_div_mul_2[grid(256)](buf24, primals_13, buf22, buf23, 256, XBLOCK=256, num_warps=4, num_stages=1)
        del primals_13
        buf25 = reinterpret_tensor(buf23, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf23  # reuse
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0[grid(16, 4)](buf18, primals_20, buf25, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        del primals_20
        buf26 = reinterpret_tensor(buf18, (16, 4, 1), (4, 1, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf25, (16, 4, 1), (4, 1, 0), 0), out=buf26)
        buf27 = reinterpret_tensor(buf22, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3[grid(16, 4)](buf26, buf27, 16, 4, XBLOCK=4, YBLOCK=16, num_warps=1, num_stages=1)
        buf28 = reinterpret_tensor(buf26, (16, 4), (4, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf27, (16, 4), (4, 1), 0), reinterpret_tensor(primals_21, (4, 4), (1, 4), 0), out=buf28)
        buf29 = reinterpret_tensor(buf28, (4, 4, 4), (16, 4, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6[grid(64)](buf29, buf15, primals_22, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_22
        buf30 = buf14; del buf14  # reuse
        buf31 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7[grid(16)](buf29, buf30, buf31, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf32 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_8[grid(64)](buf29, buf30, buf31, primals_23, primals_24, buf32, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_24
        buf33 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf32, (16, 4), (4, 1), 0), reinterpret_tensor(primals_25, (4, 4), (1, 4), 0), out=buf33)
        buf34 = reinterpret_tensor(buf33, (4, 4, 4), (16, 4, 1), 0); del buf33  # reuse
        buf40 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9[grid(64)](buf34, primals_26, buf40, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_26
        buf35 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf34, (16, 4), (4, 1), 0), reinterpret_tensor(primals_27, (4, 4), (1, 4), 0), out=buf35)
        buf36 = reinterpret_tensor(buf35, (4, 4, 4), (16, 4, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_6[grid(64)](buf36, buf32, primals_28, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del primals_28
        buf37 = buf31; del buf31  # reuse
        buf38 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_7[grid(16)](buf36, buf37, buf38, 16, XBLOCK=16, num_warps=1, num_stages=1)
        buf39 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_8[grid(64)](buf36, buf37, buf38, primals_29, primals_30, buf39, 64, XBLOCK=64, num_warps=1, num_stages=1)
        del buf37
        del buf38
        del primals_30
    return (buf39, buf8, buf24, primals_2, primals_11, primals_23, primals_29, buf8, reinterpret_tensor(buf11, (16, 4), (4, 1), 0), buf12, reinterpret_tensor(buf15, (16, 4), (4, 1), 0), reinterpret_tensor(primals_18, (16, 4), (4, 1), 0), buf24, reinterpret_tensor(buf27, (16, 4), (4, 1), 0), buf29, reinterpret_tensor(buf32, (16, 4), (4, 1), 0), reinterpret_tensor(buf34, (16, 4), (4, 1), 0), buf36, primals_27, buf40, primals_25, primals_21, reinterpret_tensor(buf25, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf19, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf20, (16, 4, 1), (4, 1, 4), 0), primals_14, primals_9, reinterpret_tensor(buf9, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf3, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf4, (16, 4, 1), (4, 1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
