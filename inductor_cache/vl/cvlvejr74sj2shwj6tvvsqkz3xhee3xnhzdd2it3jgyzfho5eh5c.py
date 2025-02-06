# AOT ID: ['2_inference']
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


# kernel path: inductor_cache/zc/czcbvzsawx47kdvn7mlrd6uvenkgurzh2xmrrld5eyru2wxhghef.py
# Topologically Sorted Source Nodes: [mul, mul_1, mse_loss, mul_2, loss, mul_3, mul_4, mse_loss_1, mul_5, loss_1, mul_6, mul_7, mse_loss_2, mul_8, loss_2, mul_9, mul_10, mse_loss_3, mul_11, loss_3, truediv], Original ATen: [aten.mul, aten.mse_loss, aten.add, aten.div]
# Source node to ATen node mapping:
#   loss => add
#   loss_1 => add_1
#   loss_2 => add_2
#   loss_3 => add_3
#   mse_loss => mean, pow_1, sub
#   mse_loss_1 => mean_1, pow_2, sub_1
#   mse_loss_2 => mean_2, pow_3, sub_2
#   mse_loss_3 => mean_3, pow_4, sub_3
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   truediv => div
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze, %select), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %select_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %mul_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_1,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, 0.5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_2, %select_2), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_3, %select_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_1, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_2,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_1, 0.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %mul_5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_4, %select_4), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_5, %select_5), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_2, 0.5), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %mul_8), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_6, %select_6), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_7, %select_7), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %mul_10), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_4,), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_3, 0.5), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_11), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_3, 4), kwargs = {})
triton_poi_fused_add_div_mse_loss_mul_0 = async_compile.triton('triton_poi_fused_add_div_mse_loss_mul_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mse_loss_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 48, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mse_loss_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp10 = tl.load(in_ptr0 + (4))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp12 = tl.load(in_ptr1 + (4))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr2 + (4))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (8))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp23 = tl.load(in_ptr1 + (8))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp26 = tl.load(in_ptr2 + (8))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp32 = tl.load(in_ptr0 + (12))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp34 = tl.load(in_ptr1 + (12))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp37 = tl.load(in_ptr2 + (12))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp49 = tl.load(in_ptr0 + (1))
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK])
    tmp51 = tl.load(in_ptr1 + (1))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp54 = tl.load(in_ptr2 + (1))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp59 = tl.load(in_ptr0 + (5))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp61 = tl.load(in_ptr1 + (5))
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp64 = tl.load(in_ptr2 + (5))
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
    tmp70 = tl.load(in_ptr0 + (9))
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK])
    tmp72 = tl.load(in_ptr1 + (9))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
    tmp75 = tl.load(in_ptr2 + (9))
    tmp76 = tl.broadcast_to(tmp75, [XBLOCK])
    tmp81 = tl.load(in_ptr0 + (13))
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK])
    tmp83 = tl.load(in_ptr1 + (13))
    tmp84 = tl.broadcast_to(tmp83, [XBLOCK])
    tmp86 = tl.load(in_ptr2 + (13))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp95 = tl.load(in_ptr0 + (2))
    tmp96 = tl.broadcast_to(tmp95, [XBLOCK])
    tmp97 = tl.load(in_ptr1 + (2))
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK])
    tmp100 = tl.load(in_ptr2 + (2))
    tmp101 = tl.broadcast_to(tmp100, [XBLOCK])
    tmp105 = tl.load(in_ptr0 + (6))
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK])
    tmp107 = tl.load(in_ptr1 + (6))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp110 = tl.load(in_ptr2 + (6))
    tmp111 = tl.broadcast_to(tmp110, [XBLOCK])
    tmp116 = tl.load(in_ptr0 + (10))
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK])
    tmp118 = tl.load(in_ptr1 + (10))
    tmp119 = tl.broadcast_to(tmp118, [XBLOCK])
    tmp121 = tl.load(in_ptr2 + (10))
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK])
    tmp127 = tl.load(in_ptr0 + (14))
    tmp128 = tl.broadcast_to(tmp127, [XBLOCK])
    tmp129 = tl.load(in_ptr1 + (14))
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK])
    tmp132 = tl.load(in_ptr2 + (14))
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK])
    tmp141 = tl.load(in_ptr0 + (3))
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK])
    tmp143 = tl.load(in_ptr1 + (3))
    tmp144 = tl.broadcast_to(tmp143, [XBLOCK])
    tmp146 = tl.load(in_ptr2 + (3))
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK])
    tmp151 = tl.load(in_ptr0 + (7))
    tmp152 = tl.broadcast_to(tmp151, [XBLOCK])
    tmp153 = tl.load(in_ptr1 + (7))
    tmp154 = tl.broadcast_to(tmp153, [XBLOCK])
    tmp156 = tl.load(in_ptr2 + (7))
    tmp157 = tl.broadcast_to(tmp156, [XBLOCK])
    tmp162 = tl.load(in_ptr0 + (11))
    tmp163 = tl.broadcast_to(tmp162, [XBLOCK])
    tmp164 = tl.load(in_ptr1 + (11))
    tmp165 = tl.broadcast_to(tmp164, [XBLOCK])
    tmp167 = tl.load(in_ptr2 + (11))
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK])
    tmp173 = tl.load(in_ptr0 + (15))
    tmp174 = tl.broadcast_to(tmp173, [XBLOCK])
    tmp175 = tl.load(in_ptr1 + (15))
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK])
    tmp178 = tl.load(in_ptr2 + (15))
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tmp7 = tmp6 * tmp3
    tmp8 = tmp4 - tmp7
    tmp9 = tmp8 * tmp8
    tmp14 = tmp11 * tmp13
    tmp17 = tmp16 * tmp13
    tmp18 = tmp14 - tmp17
    tmp19 = tmp18 * tmp18
    tmp20 = tmp9 + tmp19
    tmp25 = tmp22 * tmp24
    tmp28 = tmp27 * tmp24
    tmp29 = tmp25 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tmp20 + tmp30
    tmp36 = tmp33 * tmp35
    tmp39 = tmp38 * tmp35
    tmp40 = tmp36 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp31 + tmp41
    tmp43 = 4.0
    tmp44 = tmp42 / tmp43
    tmp45 = 0.5
    tmp46 = tmp44 * tmp45
    tmp47 = 0.0
    tmp48 = tmp46 + tmp47
    tmp53 = tmp50 * tmp52
    tmp56 = tmp55 * tmp52
    tmp57 = tmp53 - tmp56
    tmp58 = tmp57 * tmp57
    tmp63 = tmp60 * tmp62
    tmp66 = tmp65 * tmp62
    tmp67 = tmp63 - tmp66
    tmp68 = tmp67 * tmp67
    tmp69 = tmp58 + tmp68
    tmp74 = tmp71 * tmp73
    tmp77 = tmp76 * tmp73
    tmp78 = tmp74 - tmp77
    tmp79 = tmp78 * tmp78
    tmp80 = tmp69 + tmp79
    tmp85 = tmp82 * tmp84
    tmp88 = tmp87 * tmp84
    tmp89 = tmp85 - tmp88
    tmp90 = tmp89 * tmp89
    tmp91 = tmp80 + tmp90
    tmp92 = tmp91 / tmp43
    tmp93 = tmp92 * tmp45
    tmp94 = tmp48 + tmp93
    tmp99 = tmp96 * tmp98
    tmp102 = tmp101 * tmp98
    tmp103 = tmp99 - tmp102
    tmp104 = tmp103 * tmp103
    tmp109 = tmp106 * tmp108
    tmp112 = tmp111 * tmp108
    tmp113 = tmp109 - tmp112
    tmp114 = tmp113 * tmp113
    tmp115 = tmp104 + tmp114
    tmp120 = tmp117 * tmp119
    tmp123 = tmp122 * tmp119
    tmp124 = tmp120 - tmp123
    tmp125 = tmp124 * tmp124
    tmp126 = tmp115 + tmp125
    tmp131 = tmp128 * tmp130
    tmp134 = tmp133 * tmp130
    tmp135 = tmp131 - tmp134
    tmp136 = tmp135 * tmp135
    tmp137 = tmp126 + tmp136
    tmp138 = tmp137 / tmp43
    tmp139 = tmp138 * tmp45
    tmp140 = tmp94 + tmp139
    tmp145 = tmp142 * tmp144
    tmp148 = tmp147 * tmp144
    tmp149 = tmp145 - tmp148
    tmp150 = tmp149 * tmp149
    tmp155 = tmp152 * tmp154
    tmp158 = tmp157 * tmp154
    tmp159 = tmp155 - tmp158
    tmp160 = tmp159 * tmp159
    tmp161 = tmp150 + tmp160
    tmp166 = tmp163 * tmp165
    tmp169 = tmp168 * tmp165
    tmp170 = tmp166 - tmp169
    tmp171 = tmp170 * tmp170
    tmp172 = tmp161 + tmp171
    tmp177 = tmp174 * tmp176
    tmp180 = tmp179 * tmp176
    tmp181 = tmp177 - tmp180
    tmp182 = tmp181 * tmp181
    tmp183 = tmp172 + tmp182
    tmp184 = tmp183 / tmp43
    tmp185 = tmp184 * tmp45
    tmp186 = tmp140 + tmp185
    tmp187 = 0.25
    tmp188 = tmp186 * tmp187
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp188, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4), (4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    assert_size_stride(arg2_1, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [mul, mul_1, mse_loss, mul_2, loss, mul_3, mul_4, mse_loss_1, mul_5, loss_1, mul_6, mul_7, mse_loss_2, mul_8, loss_2, mul_9, mul_10, mse_loss_3, mul_11, loss_3, truediv], Original ATen: [aten.mul, aten.mse_loss, aten.add, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mse_loss_mul_0.run(buf1, arg0_1, arg2_1, arg1_1, 1, grid=grid(1), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
