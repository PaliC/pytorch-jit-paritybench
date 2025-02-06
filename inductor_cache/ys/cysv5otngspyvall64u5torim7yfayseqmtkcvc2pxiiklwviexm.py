# AOT ID: ['1_inference']
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


# kernel path: inductor_cache/7d/c7dk6yzot5g3ndp3emerxu33zct4ceevmcpma7d6xx5zufsf7g4r.py
# Topologically Sorted Source Nodes: [mul, sin, mul_1, sin_1, mul_2, sin_2, mul_3, sin_3, mul_4, sin_4, mul_5, sin_5, mul_6, sin_6, mul_7, sin_7, mul_8, sin_8, mul_9, sin_9, mul_10, sin_10, mul_11, sin_11, mul_12, sin_12, mul_13, sin_13, mul_14, sin_14, mul_15, sin_15, mul_16, sin_16, mul_17, sin_17, mul_18, sin_18, mul_19, sin_19, mul_20, cos, mul_21, cos_1, mul_22, cos_2, mul_23, cos_3, mul_24, cos_4, mul_25, cos_5, mul_26, cos_6, mul_27, cos_7, mul_28, cos_8, mul_29, cos_9, mul_30, cos_10, mul_31, cos_11, mul_32, cos_12, mul_33, cos_13, mul_34, cos_14, mul_35, cos_15, mul_36, cos_16, mul_37, cos_17, mul_38, cos_18, mul_39, cos_19, cat], Original ATen: [aten.mul, aten.sin, aten.cos, aten.cat]
# Source node to ATen node mapping:
#   cat => cat
#   cos => cos
#   cos_1 => cos_1
#   cos_10 => cos_10
#   cos_11 => cos_11
#   cos_12 => cos_12
#   cos_13 => cos_13
#   cos_14 => cos_14
#   cos_15 => cos_15
#   cos_16 => cos_16
#   cos_17 => cos_17
#   cos_18 => cos_18
#   cos_19 => cos_19
#   cos_2 => cos_2
#   cos_3 => cos_3
#   cos_4 => cos_4
#   cos_5 => cos_5
#   cos_6 => cos_6
#   cos_7 => cos_7
#   cos_8 => cos_8
#   cos_9 => cos_9
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_21 => mul_21
#   mul_22 => mul_22
#   mul_23 => mul_23
#   mul_24 => mul_24
#   mul_25 => mul_25
#   mul_26 => mul_26
#   mul_27 => mul_27
#   mul_28 => mul_28
#   mul_29 => mul_29
#   mul_3 => mul_3
#   mul_30 => mul_30
#   mul_31 => mul_31
#   mul_32 => mul_32
#   mul_33 => mul_33
#   mul_34 => mul_34
#   mul_35 => mul_35
#   mul_36 => mul_36
#   mul_37 => mul_37
#   mul_38 => mul_38
#   mul_39 => mul_39
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   sin => sin
#   sin_1 => sin_1
#   sin_10 => sin_10
#   sin_11 => sin_11
#   sin_12 => sin_12
#   sin_13 => sin_13
#   sin_14 => sin_14
#   sin_15 => sin_15
#   sin_16 => sin_16
#   sin_17 => sin_17
#   sin_18 => sin_18
#   sin_19 => sin_19
#   sin_2 => sin_2
#   sin_3 => sin_3
#   sin_4 => sin_4
#   sin_5 => sin_5
#   sin_6 => sin_6
#   sin_7 => sin_7
#   sin_8 => sin_8
#   sin_9 => sin_9
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_1), kwargs = {})
#   %sin_1 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_2), kwargs = {})
#   %sin_2 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_3), kwargs = {})
#   %sin_3 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_3,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_4), kwargs = {})
#   %sin_4 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_5), kwargs = {})
#   %sin_5 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_5,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_6), kwargs = {})
#   %sin_6 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_6,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_7), kwargs = {})
#   %sin_7 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_7,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_8), kwargs = {})
#   %sin_8 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_8,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_9), kwargs = {})
#   %sin_9 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_9,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_10), kwargs = {})
#   %sin_10 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_10,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_11), kwargs = {})
#   %sin_11 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_11,), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_12), kwargs = {})
#   %sin_12 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_12,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_13), kwargs = {})
#   %sin_13 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_13,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_14), kwargs = {})
#   %sin_14 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_14,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_15), kwargs = {})
#   %sin_15 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_15,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_16), kwargs = {})
#   %sin_16 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_16,), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_17), kwargs = {})
#   %sin_17 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_17,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_18), kwargs = {})
#   %sin_18 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_18,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_19), kwargs = {})
#   %sin_19 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_19,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_20), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_20,), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_21), kwargs = {})
#   %cos_1 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_21,), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_22), kwargs = {})
#   %cos_2 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_22,), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_23), kwargs = {})
#   %cos_3 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_23,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_24), kwargs = {})
#   %cos_4 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_24,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_25), kwargs = {})
#   %cos_5 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_25,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_26), kwargs = {})
#   %cos_6 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_26,), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_27), kwargs = {})
#   %cos_7 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_27,), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_28), kwargs = {})
#   %cos_8 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_28,), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_29), kwargs = {})
#   %cos_9 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_29,), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_30), kwargs = {})
#   %cos_10 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_30,), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_31), kwargs = {})
#   %cos_11 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_31,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_32), kwargs = {})
#   %cos_12 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_32,), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_33), kwargs = {})
#   %cos_13 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_33,), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_34), kwargs = {})
#   %cos_14 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_34,), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_35), kwargs = {})
#   %cos_15 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_35,), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_36), kwargs = {})
#   %cos_16 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_36,), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_37), kwargs = {})
#   %cos_17 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_37,), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_38), kwargs = {})
#   %cos_18 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_38,), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, %select_39), kwargs = {})
#   %cos_19 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_39,), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg1_1, %sin, %sin_1, %sin_2, %sin_3, %sin_4, %sin_5, %sin_6, %sin_7, %sin_8, %sin_9, %sin_10, %sin_11, %sin_12, %sin_13, %sin_14, %sin_15, %sin_16, %sin_17, %sin_18, %sin_19, %cos, %cos_1, %cos_2, %cos_3, %cos_4, %cos_5, %cos_6, %cos_7, %cos_8, %cos_9, %cos_10, %cos_11, %cos_12, %cos_13, %cos_14, %cos_15, %cos_16, %cos_17, %cos_18, %cos_19], 1), kwargs = {})
triton_poi_fused_cat_cos_mul_sin_0 = async_compile.triton('triton_poi_fused_cat_cos_mul_sin_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'out_ptr12': '*fp32', 'out_ptr13': '*fp32', 'out_ptr14': '*fp32', 'out_ptr15': '*fp32', 'out_ptr16': '*fp32', 'out_ptr17': '*fp32', 'out_ptr18': '*fp32', 'out_ptr19': '*fp32', 'out_ptr20': '*fp32', 'out_ptr21': '*fp32', 'out_ptr22': '*fp32', 'out_ptr23': '*fp32', 'out_ptr24': '*fp32', 'out_ptr25': '*fp32', 'out_ptr26': '*fp32', 'out_ptr27': '*fp32', 'out_ptr28': '*fp32', 'out_ptr29': '*fp32', 'out_ptr30': '*fp32', 'out_ptr31': '*fp32', 'out_ptr32': '*fp32', 'out_ptr33': '*fp32', 'out_ptr34': '*fp32', 'out_ptr35': '*fp32', 'out_ptr36': '*fp32', 'out_ptr37': '*fp32', 'out_ptr38': '*fp32', 'out_ptr39': '*fp32', 'out_ptr40': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_cos_mul_sin_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 21, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_cos_mul_sin_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + (3))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp21 = tl.load(in_ptr1 + (4))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (5))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp31 = tl.load(in_ptr1 + (6))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp36 = tl.load(in_ptr1 + (7))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp41 = tl.load(in_ptr1 + (8))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp46 = tl.load(in_ptr1 + (9))
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK])
    tmp51 = tl.load(in_ptr1 + (10))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp56 = tl.load(in_ptr1 + (11))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp61 = tl.load(in_ptr1 + (12))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp66 = tl.load(in_ptr1 + (13))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp71 = tl.load(in_ptr1 + (14))
    tmp72 = tl.broadcast_to(tmp71, [XBLOCK])
    tmp76 = tl.load(in_ptr1 + (15))
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
    tmp81 = tl.load(in_ptr1 + (16))
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK])
    tmp86 = tl.load(in_ptr1 + (17))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp91 = tl.load(in_ptr1 + (18))
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK])
    tmp96 = tl.load(in_ptr1 + (19))
    tmp97 = tl.broadcast_to(tmp96, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp4 = tl_math.sin(tmp3)
    tmp5 = tl_math.cos(tmp3)
    tmp8 = tmp0 * tmp7
    tmp9 = tl_math.sin(tmp8)
    tmp10 = tl_math.cos(tmp8)
    tmp13 = tmp0 * tmp12
    tmp14 = tl_math.sin(tmp13)
    tmp15 = tl_math.cos(tmp13)
    tmp18 = tmp0 * tmp17
    tmp19 = tl_math.sin(tmp18)
    tmp20 = tl_math.cos(tmp18)
    tmp23 = tmp0 * tmp22
    tmp24 = tl_math.sin(tmp23)
    tmp25 = tl_math.cos(tmp23)
    tmp28 = tmp0 * tmp27
    tmp29 = tl_math.sin(tmp28)
    tmp30 = tl_math.cos(tmp28)
    tmp33 = tmp0 * tmp32
    tmp34 = tl_math.sin(tmp33)
    tmp35 = tl_math.cos(tmp33)
    tmp38 = tmp0 * tmp37
    tmp39 = tl_math.sin(tmp38)
    tmp40 = tl_math.cos(tmp38)
    tmp43 = tmp0 * tmp42
    tmp44 = tl_math.sin(tmp43)
    tmp45 = tl_math.cos(tmp43)
    tmp48 = tmp0 * tmp47
    tmp49 = tl_math.sin(tmp48)
    tmp50 = tl_math.cos(tmp48)
    tmp53 = tmp0 * tmp52
    tmp54 = tl_math.sin(tmp53)
    tmp55 = tl_math.cos(tmp53)
    tmp58 = tmp0 * tmp57
    tmp59 = tl_math.sin(tmp58)
    tmp60 = tl_math.cos(tmp58)
    tmp63 = tmp0 * tmp62
    tmp64 = tl_math.sin(tmp63)
    tmp65 = tl_math.cos(tmp63)
    tmp68 = tmp0 * tmp67
    tmp69 = tl_math.sin(tmp68)
    tmp70 = tl_math.cos(tmp68)
    tmp73 = tmp0 * tmp72
    tmp74 = tl_math.sin(tmp73)
    tmp75 = tl_math.cos(tmp73)
    tmp78 = tmp0 * tmp77
    tmp79 = tl_math.sin(tmp78)
    tmp80 = tl_math.cos(tmp78)
    tmp83 = tmp0 * tmp82
    tmp84 = tl_math.sin(tmp83)
    tmp85 = tl_math.cos(tmp83)
    tmp88 = tmp0 * tmp87
    tmp89 = tl_math.sin(tmp88)
    tmp90 = tl_math.cos(tmp88)
    tmp93 = tmp0 * tmp92
    tmp94 = tl_math.sin(tmp93)
    tmp95 = tl_math.cos(tmp93)
    tmp98 = tmp0 * tmp97
    tmp99 = tl_math.sin(tmp98)
    tmp100 = tl_math.cos(tmp98)
    tl.store(out_ptr0 + (x0 + 2624*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 2624*x1), tmp4, xmask)
    tl.store(out_ptr2 + (x0 + 2624*x1), tmp5, xmask)
    tl.store(out_ptr3 + (x0 + 2624*x1), tmp9, xmask)
    tl.store(out_ptr4 + (x0 + 2624*x1), tmp10, xmask)
    tl.store(out_ptr5 + (x0 + 2624*x1), tmp14, xmask)
    tl.store(out_ptr6 + (x0 + 2624*x1), tmp15, xmask)
    tl.store(out_ptr7 + (x0 + 2624*x1), tmp19, xmask)
    tl.store(out_ptr8 + (x0 + 2624*x1), tmp20, xmask)
    tl.store(out_ptr9 + (x0 + 2624*x1), tmp24, xmask)
    tl.store(out_ptr10 + (x0 + 2624*x1), tmp25, xmask)
    tl.store(out_ptr11 + (x0 + 2624*x1), tmp29, xmask)
    tl.store(out_ptr12 + (x0 + 2624*x1), tmp30, xmask)
    tl.store(out_ptr13 + (x0 + 2624*x1), tmp34, xmask)
    tl.store(out_ptr14 + (x0 + 2624*x1), tmp35, xmask)
    tl.store(out_ptr15 + (x0 + 2624*x1), tmp39, xmask)
    tl.store(out_ptr16 + (x0 + 2624*x1), tmp40, xmask)
    tl.store(out_ptr17 + (x0 + 2624*x1), tmp44, xmask)
    tl.store(out_ptr18 + (x0 + 2624*x1), tmp45, xmask)
    tl.store(out_ptr19 + (x0 + 2624*x1), tmp49, xmask)
    tl.store(out_ptr20 + (x0 + 2624*x1), tmp50, xmask)
    tl.store(out_ptr21 + (x0 + 2624*x1), tmp54, xmask)
    tl.store(out_ptr22 + (x0 + 2624*x1), tmp55, xmask)
    tl.store(out_ptr23 + (x0 + 2624*x1), tmp59, xmask)
    tl.store(out_ptr24 + (x0 + 2624*x1), tmp60, xmask)
    tl.store(out_ptr25 + (x0 + 2624*x1), tmp64, xmask)
    tl.store(out_ptr26 + (x0 + 2624*x1), tmp65, xmask)
    tl.store(out_ptr27 + (x0 + 2624*x1), tmp69, xmask)
    tl.store(out_ptr28 + (x0 + 2624*x1), tmp70, xmask)
    tl.store(out_ptr29 + (x0 + 2624*x1), tmp74, xmask)
    tl.store(out_ptr30 + (x0 + 2624*x1), tmp75, xmask)
    tl.store(out_ptr31 + (x0 + 2624*x1), tmp79, xmask)
    tl.store(out_ptr32 + (x0 + 2624*x1), tmp80, xmask)
    tl.store(out_ptr33 + (x0 + 2624*x1), tmp84, xmask)
    tl.store(out_ptr34 + (x0 + 2624*x1), tmp85, xmask)
    tl.store(out_ptr35 + (x0 + 2624*x1), tmp89, xmask)
    tl.store(out_ptr36 + (x0 + 2624*x1), tmp90, xmask)
    tl.store(out_ptr37 + (x0 + 2624*x1), tmp94, xmask)
    tl.store(out_ptr38 + (x0 + 2624*x1), tmp95, xmask)
    tl.store(out_ptr39 + (x0 + 2624*x1), tmp99, xmask)
    tl.store(out_ptr40 + (x0 + 2624*x1), tmp100, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (20, ), (1, ))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf41 = empty_strided_cuda((4, 164, 4, 4), (2624, 16, 4, 1), torch.float32)
        buf0 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 0)  # alias
        buf1 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 64)  # alias
        buf21 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1344)  # alias
        buf2 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 128)  # alias
        buf22 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1408)  # alias
        buf3 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 192)  # alias
        buf23 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1472)  # alias
        buf4 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 256)  # alias
        buf24 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1536)  # alias
        buf5 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 320)  # alias
        buf25 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1600)  # alias
        buf6 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 384)  # alias
        buf26 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1664)  # alias
        buf7 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 448)  # alias
        buf27 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1728)  # alias
        buf8 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 512)  # alias
        buf28 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1792)  # alias
        buf9 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 576)  # alias
        buf29 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1856)  # alias
        buf10 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 640)  # alias
        buf30 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1920)  # alias
        buf11 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 704)  # alias
        buf31 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1984)  # alias
        buf12 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 768)  # alias
        buf32 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2048)  # alias
        buf13 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 832)  # alias
        buf33 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2112)  # alias
        buf14 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 896)  # alias
        buf34 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2176)  # alias
        buf15 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 960)  # alias
        buf35 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2240)  # alias
        buf16 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1024)  # alias
        buf36 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2304)  # alias
        buf17 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1088)  # alias
        buf37 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2368)  # alias
        buf18 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1152)  # alias
        buf38 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2432)  # alias
        buf19 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1216)  # alias
        buf39 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2496)  # alias
        buf20 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 1280)  # alias
        buf40 = reinterpret_tensor(buf41, (4, 4, 4, 4), (2624, 16, 4, 1), 2560)  # alias
        # Topologically Sorted Source Nodes: [mul, sin, mul_1, sin_1, mul_2, sin_2, mul_3, sin_3, mul_4, sin_4, mul_5, sin_5, mul_6, sin_6, mul_7, sin_7, mul_8, sin_8, mul_9, sin_9, mul_10, sin_10, mul_11, sin_11, mul_12, sin_12, mul_13, sin_13, mul_14, sin_14, mul_15, sin_15, mul_16, sin_16, mul_17, sin_17, mul_18, sin_18, mul_19, sin_19, mul_20, cos, mul_21, cos_1, mul_22, cos_2, mul_23, cos_3, mul_24, cos_4, mul_25, cos_5, mul_26, cos_6, mul_27, cos_7, mul_28, cos_8, mul_29, cos_9, mul_30, cos_10, mul_31, cos_11, mul_32, cos_12, mul_33, cos_13, mul_34, cos_14, mul_35, cos_15, mul_36, cos_16, mul_37, cos_17, mul_38, cos_18, mul_39, cos_19, cat], Original ATen: [aten.mul, aten.sin, aten.cos, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_cos_mul_sin_0.run(arg1_1, arg0_1, buf0, buf1, buf21, buf2, buf22, buf3, buf23, buf4, buf24, buf5, buf25, buf6, buf26, buf7, buf27, buf8, buf28, buf9, buf29, buf10, buf30, buf11, buf31, buf12, buf32, buf13, buf33, buf14, buf34, buf15, buf35, buf16, buf36, buf17, buf37, buf18, buf38, buf19, buf39, buf20, buf40, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf41, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
