# AOT ID: ['10_inference']
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


# kernel path: inductor_cache/t7/ct7kdvtjaw2sumtdsxpoyucb36exxayuetyyhehpm5hjxgn6szue.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/vc/cvcerakyysng7n3pg6qonj5kxgpcyv2ku6rxlwpm5dwi3usfkbhk.py
# Topologically Sorted Source Nodes: [mu_x, mul_2, mu_y, mul_3, add, mul, avg_pool2d_4, mul_1, sigma_xy, mul_4, add_1, SSIM_n, pow_5, pow_6, add_2, add_3, pow_1, avg_pool2d_2, pow_2, sigma_x, pow_3, avg_pool2d_3, pow_4, sigma_y, add_4, add_5, SSIM_d, truediv, sub_3, truediv_1, clamp], Original ATen: [aten.avg_pool2d, aten.mul, aten.add, aten.sub, aten.pow, aten.div, aten.rsub, aten.clamp]
# Source node to ATen node mapping:
#   SSIM_d => mul_6
#   SSIM_n => mul_5
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   avg_pool2d_2 => avg_pool2d_2
#   avg_pool2d_3 => avg_pool2d_3
#   avg_pool2d_4 => avg_pool2d_4
#   clamp => clamp_max, clamp_min
#   mu_x => avg_pool2d
#   mu_y => avg_pool2d_1
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   sigma_x => sub
#   sigma_xy => sub_2
#   sigma_y => sub_1
#   sub_3 => sub_3
#   truediv => div
#   truediv_1 => div_1
# Graph fragment:
#   %avg_pool2d : [num_users=4] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%arg0_1, [1, 9], [1, 1]), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%avg_pool2d, 2), kwargs = {})
#   %avg_pool2d_1 : [num_users=4] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%arg1_1, [1, 9], [1, 1]), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %avg_pool2d_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 0.0001), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
#   %avg_pool2d_4 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%mul, [1, 9], [1, 1]), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%avg_pool2d, %avg_pool2d_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_4, %mul_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0.0009), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %add_1), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%avg_pool2d, 2), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%avg_pool2d_1, 2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_5, %pow_6), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0.0001), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_1, [1, 9], [1, 1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%avg_pool2d, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_2, %pow_2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg1_1, 2), kwargs = {})
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%pow_3, [1, 9], [1, 1]), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%avg_pool2d_1, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%avg_pool2d_3, %pow_4), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %sub_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, 0.0009), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %add_5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %div), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_3, 2), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_1, 0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1), kwargs = {})
triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_rsub_sub_1 = async_compile.triton('triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_rsub_sub_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_rsub_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 27, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_rsub_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 56)
    x1 = xindex // 56
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), None)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 64*x1), None)
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 64*x1), None)
    tmp5 = tl.load(in_ptr0 + (3 + x0 + 64*x1), None)
    tmp7 = tl.load(in_ptr0 + (4 + x0 + 64*x1), None)
    tmp9 = tl.load(in_ptr0 + (5 + x0 + 64*x1), None)
    tmp11 = tl.load(in_ptr0 + (6 + x0 + 64*x1), None)
    tmp13 = tl.load(in_ptr0 + (7 + x0 + 64*x1), None)
    tmp15 = tl.load(in_ptr0 + (8 + x0 + 64*x1), None)
    tmp19 = tl.load(in_ptr1 + (x0 + 64*x1), None)
    tmp20 = tl.load(in_ptr1 + (1 + x0 + 64*x1), None)
    tmp22 = tl.load(in_ptr1 + (2 + x0 + 64*x1), None)
    tmp24 = tl.load(in_ptr1 + (3 + x0 + 64*x1), None)
    tmp26 = tl.load(in_ptr1 + (4 + x0 + 64*x1), None)
    tmp28 = tl.load(in_ptr1 + (5 + x0 + 64*x1), None)
    tmp30 = tl.load(in_ptr1 + (6 + x0 + 64*x1), None)
    tmp32 = tl.load(in_ptr1 + (7 + x0 + 64*x1), None)
    tmp34 = tl.load(in_ptr1 + (8 + x0 + 64*x1), None)
    tmp55 = tl.load(in_ptr2 + (x0 + 64*x1), None)
    tmp56 = tl.load(in_ptr2 + (1 + x0 + 64*x1), None)
    tmp58 = tl.load(in_ptr2 + (2 + x0 + 64*x1), None)
    tmp60 = tl.load(in_ptr2 + (3 + x0 + 64*x1), None)
    tmp62 = tl.load(in_ptr2 + (4 + x0 + 64*x1), None)
    tmp64 = tl.load(in_ptr2 + (5 + x0 + 64*x1), None)
    tmp66 = tl.load(in_ptr2 + (6 + x0 + 64*x1), None)
    tmp68 = tl.load(in_ptr2 + (7 + x0 + 64*x1), None)
    tmp70 = tl.load(in_ptr2 + (8 + x0 + 64*x1), None)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp17 = 0.1111111111111111
    tmp18 = tmp16 * tmp17
    tmp21 = tmp20 + tmp19
    tmp23 = tmp22 + tmp21
    tmp25 = tmp24 + tmp23
    tmp27 = tmp26 + tmp25
    tmp29 = tmp28 + tmp27
    tmp31 = tmp30 + tmp29
    tmp33 = tmp32 + tmp31
    tmp35 = tmp34 + tmp33
    tmp36 = tmp35 * tmp17
    tmp37 = tmp19 * tmp19
    tmp38 = tmp20 * tmp20
    tmp39 = tmp38 + tmp37
    tmp40 = tmp22 * tmp22
    tmp41 = tmp40 + tmp39
    tmp42 = tmp24 * tmp24
    tmp43 = tmp42 + tmp41
    tmp44 = tmp26 * tmp26
    tmp45 = tmp44 + tmp43
    tmp46 = tmp28 * tmp28
    tmp47 = tmp46 + tmp45
    tmp48 = tmp30 * tmp30
    tmp49 = tmp48 + tmp47
    tmp50 = tmp32 * tmp32
    tmp51 = tmp50 + tmp49
    tmp52 = tmp34 * tmp34
    tmp53 = tmp52 + tmp51
    tmp54 = tmp53 * tmp17
    tmp57 = tmp56 + tmp55
    tmp59 = tmp58 + tmp57
    tmp61 = tmp60 + tmp59
    tmp63 = tmp62 + tmp61
    tmp65 = tmp64 + tmp63
    tmp67 = tmp66 + tmp65
    tmp69 = tmp68 + tmp67
    tmp71 = tmp70 + tmp69
    tmp72 = tmp71 * tmp17
    tmp73 = tmp55 * tmp55
    tmp74 = tmp56 * tmp56
    tmp75 = tmp74 + tmp73
    tmp76 = tmp58 * tmp58
    tmp77 = tmp76 + tmp75
    tmp78 = tmp60 * tmp60
    tmp79 = tmp78 + tmp77
    tmp80 = tmp62 * tmp62
    tmp81 = tmp80 + tmp79
    tmp82 = tmp64 * tmp64
    tmp83 = tmp82 + tmp81
    tmp84 = tmp66 * tmp66
    tmp85 = tmp84 + tmp83
    tmp86 = tmp68 * tmp68
    tmp87 = tmp86 + tmp85
    tmp88 = tmp70 * tmp70
    tmp89 = tmp88 + tmp87
    tmp90 = tmp89 * tmp17
    tmp91 = 2.0
    tmp92 = tmp36 * tmp91
    tmp93 = tmp92 * tmp72
    tmp94 = 0.0001
    tmp95 = tmp93 + tmp94
    tmp96 = tmp36 * tmp72
    tmp97 = tmp18 - tmp96
    tmp98 = tmp97 * tmp91
    tmp99 = 0.0009
    tmp100 = tmp98 + tmp99
    tmp101 = tmp95 * tmp100
    tmp102 = tmp36 * tmp36
    tmp103 = tmp72 * tmp72
    tmp104 = tmp102 + tmp103
    tmp105 = tmp104 + tmp94
    tmp106 = tmp54 - tmp102
    tmp107 = tmp90 - tmp103
    tmp108 = tmp106 + tmp107
    tmp109 = tmp108 + tmp99
    tmp110 = tmp105 * tmp109
    tmp111 = tmp101 / tmp110
    tmp112 = 1.0
    tmp113 = tmp112 - tmp111
    tmp114 = 0.5
    tmp115 = tmp113 * tmp114
    tmp116 = 0.0
    tmp117 = triton_helpers.maximum(tmp115, tmp116)
    tmp118 = triton_helpers.minimum(tmp117, tmp112)
    tl.store(in_out_ptr0 + (x2), tmp118, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(arg1_1, (4, 4, 64, 64), (16384, 4096, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((4, 4, 64, 64), (16384, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(arg0_1, arg1_1, buf2, 65536, grid=grid(65536), stream=stream0)
        buf0 = empty_strided_cuda((4, 4, 64, 56), (14336, 3584, 56, 1), torch.float32)
        buf6 = buf0; del buf0  # reuse
        buf7 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [mu_x, mul_2, mu_y, mul_3, add, mul, avg_pool2d_4, mul_1, sigma_xy, mul_4, add_1, SSIM_n, pow_5, pow_6, add_2, add_3, pow_1, avg_pool2d_2, pow_2, sigma_x, pow_3, avg_pool2d_3, pow_4, sigma_y, add_4, add_5, SSIM_d, truediv, sub_3, truediv_1, clamp], Original ATen: [aten.avg_pool2d, aten.mul, aten.add, aten.sub, aten.pow, aten.div, aten.rsub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_rsub_sub_1.run(buf7, buf2, arg0_1, arg1_1, 57344, grid=grid(57344), stream=stream0)
        del arg0_1
        del arg1_1
        del buf2
    return (buf7, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
