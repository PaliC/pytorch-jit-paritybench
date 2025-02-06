# AOT ID: ['0_inference']
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


# kernel path: inductor_cache/xx/cxxkxc63gz3xfocjzan7zw4xyvdte3p27ams5wlkuzqqzsbszze7.py
# Topologically Sorted Source Nodes: [mul, alpha, sub, clamp, mul_1, cos, sub_1, w, mul_2, sin, mul_3, mul_4, cos_1, mul_5, sub_2, clamp_1, mul_6, cos_2, sub_3, w_1, mul_7, sin_1, mul_8, mul_9, cos_3, mul_10, sub_4, clamp_2, mul_11, cos_4, sub_5, w_2, mul_12, sin_2, mul_13, mul_14, cos_5, mul_15, sub_6, clamp_3, mul_16, cos_6, sub_7, w_3, mul_17, sin_3, mul_18, mul_19, cos_7, mul_20, cat], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.cos, aten.rsub, aten.sin, aten.cat]
# Source node to ATen node mapping:
#   alpha => div
#   cat => cat
#   clamp => clamp_max, clamp_min
#   clamp_1 => clamp_max_1, clamp_min_1
#   clamp_2 => clamp_max_2, clamp_min_2
#   clamp_3 => clamp_max_3, clamp_min_3
#   cos => cos
#   cos_1 => cos_1
#   cos_2 => cos_2
#   cos_3 => cos_3
#   cos_4 => cos_4
#   cos_5 => cos_5
#   cos_6 => cos_6
#   cos_7 => cos_7
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
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   sin => sin
#   sin_1 => sin_1
#   sin_2 => sin_2
#   sin_3 => sin_3
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   w => div_1
#   w_1 => div_2
#   w_2 => div_3
#   w_3 => div_4
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 4), kwargs = {})
#   %div : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul, 4.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %select_4), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 3.141592653589793), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %cos), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, 2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %arg3_1), kwargs = {})
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %sin), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %arg3_1), kwargs = {})
#   %cos_1 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_4,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %cos_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %select_5), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 3.141592653589793), kwargs = {})
#   %cos_2 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_6,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %cos_2), kwargs = {})
#   %div_2 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, 2), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %arg3_1), kwargs = {})
#   %sin_1 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_7,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %sin_1), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %arg3_1), kwargs = {})
#   %cos_3 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_9,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %cos_3), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %select_6), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_4, 0), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_2, 3.141592653589793), kwargs = {})
#   %cos_4 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_11,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %cos_4), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, 2), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %arg3_1), kwargs = {})
#   %sin_2 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_12,), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %sin_2), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %arg3_1), kwargs = {})
#   %cos_5 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_14,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %cos_5), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div, %select_7), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_6, 0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_3, 3.141592653589793), kwargs = {})
#   %cos_6 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_16,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %cos_6), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_7, 2), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %arg3_1), kwargs = {})
#   %sin_3 : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mul_17,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %sin_3), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %arg3_1), kwargs = {})
#   %cos_7 : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%mul_19,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, %cos_7), kwargs = {})
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%arg3_1, %mul_3, %mul_5, %mul_8, %mul_10, %mul_13, %mul_15, %mul_18, %mul_20], -1), kwargs = {})
triton_poi_fused_cat_clamp_cos_div_mul_rsub_sin_sub_0 = async_compile.triton('triton_poi_fused_cat_clamp_cos_div_mul_rsub_sin_sub_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 4, 8, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_clamp_cos_div_mul_rsub_sin_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_clamp_cos_div_mul_rsub_sin_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp26 = tl.load(in_ptr2 + (1))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp35 = tl.load(in_ptr3 + (1))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp42 = tl.load(in_ptr2 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp51 = tl.load(in_ptr3 + (2))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp58 = tl.load(in_ptr2 + (3))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp67 = tl.load(in_ptr3 + (3))
    tmp68 = tl.broadcast_to(tmp67, [XBLOCK])
    tmp2 = 4.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.25
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 - tmp7
    tmp9 = 0.0
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = 1.0
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = 3.141592653589793
    tmp14 = tmp12 * tmp13
    tmp15 = tl_math.cos(tmp14)
    tmp16 = tmp11 - tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp21 = tmp20 * tmp0
    tmp22 = tl_math.sin(tmp21)
    tmp23 = tmp18 * tmp22
    tmp24 = tl_math.cos(tmp21)
    tmp25 = tmp18 * tmp24
    tmp28 = tmp5 - tmp27
    tmp29 = triton_helpers.maximum(tmp28, tmp9)
    tmp30 = triton_helpers.minimum(tmp29, tmp11)
    tmp31 = tmp30 * tmp13
    tmp32 = tl_math.cos(tmp31)
    tmp33 = tmp11 - tmp32
    tmp34 = tmp33 * tmp17
    tmp37 = tmp36 * tmp0
    tmp38 = tl_math.sin(tmp37)
    tmp39 = tmp34 * tmp38
    tmp40 = tl_math.cos(tmp37)
    tmp41 = tmp34 * tmp40
    tmp44 = tmp5 - tmp43
    tmp45 = triton_helpers.maximum(tmp44, tmp9)
    tmp46 = triton_helpers.minimum(tmp45, tmp11)
    tmp47 = tmp46 * tmp13
    tmp48 = tl_math.cos(tmp47)
    tmp49 = tmp11 - tmp48
    tmp50 = tmp49 * tmp17
    tmp53 = tmp52 * tmp0
    tmp54 = tl_math.sin(tmp53)
    tmp55 = tmp50 * tmp54
    tmp56 = tl_math.cos(tmp53)
    tmp57 = tmp50 * tmp56
    tmp60 = tmp5 - tmp59
    tmp61 = triton_helpers.maximum(tmp60, tmp9)
    tmp62 = triton_helpers.minimum(tmp61, tmp11)
    tmp63 = tmp62 * tmp13
    tmp64 = tl_math.cos(tmp63)
    tmp65 = tmp11 - tmp64
    tmp66 = tmp65 * tmp17
    tmp69 = tmp68 * tmp0
    tmp70 = tl_math.sin(tmp69)
    tmp71 = tmp66 * tmp70
    tmp72 = tl_math.cos(tmp69)
    tmp73 = tmp66 * tmp72
    tl.store(out_ptr0 + (x0 + 36*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 36*x1), tmp23, xmask)
    tl.store(out_ptr2 + (x0 + 36*x1), tmp25, xmask)
    tl.store(out_ptr3 + (x0 + 36*x1), tmp39, xmask)
    tl.store(out_ptr4 + (x0 + 36*x1), tmp41, xmask)
    tl.store(out_ptr5 + (x0 + 36*x1), tmp55, xmask)
    tl.store(out_ptr6 + (x0 + 36*x1), tmp57, xmask)
    tl.store(out_ptr7 + (x0 + 36*x1), tmp71, xmask)
    tl.store(out_ptr8 + (x0 + 36*x1), tmp73, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, ), (1, ))
    assert_size_stride(arg2_1, (4, ), (1, ))
    assert_size_stride(arg3_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf9 = empty_strided_cuda((4, 4, 4, 36), (576, 144, 36, 1), torch.float32)
        buf0 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 0)  # alias
        buf1 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 4)  # alias
        buf2 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 8)  # alias
        buf3 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 12)  # alias
        buf4 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 16)  # alias
        buf5 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 20)  # alias
        buf6 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 24)  # alias
        buf7 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 28)  # alias
        buf8 = reinterpret_tensor(buf9, (4, 4, 4, 4), (576, 144, 36, 1), 32)  # alias
        # Topologically Sorted Source Nodes: [mul, alpha, sub, clamp, mul_1, cos, sub_1, w, mul_2, sin, mul_3, mul_4, cos_1, mul_5, sub_2, clamp_1, mul_6, cos_2, sub_3, w_1, mul_7, sin_1, mul_8, mul_9, cos_3, mul_10, sub_4, clamp_2, mul_11, cos_4, sub_5, w_2, mul_12, sin_2, mul_13, mul_14, cos_5, mul_15, sub_6, clamp_3, mul_16, cos_6, sub_7, w_3, mul_17, sin_3, mul_18, mul_19, cos_7, mul_20, cat], Original ATen: [aten.mul, aten.div, aten.sub, aten.clamp, aten.cos, aten.rsub, aten.sin, aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_clamp_cos_div_mul_rsub_sin_sub_0.run(arg3_1, arg0_1, arg2_1, arg1_1, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, 256, grid=grid(256), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
