# AOT ID: ['12_inference']
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


# kernel path: inductor_cache/7l/c7ll46rptlpsgjncwhhzhn23hlylrpctek3vj27v63ctfodbpvqa.py
# Topologically Sorted Source Nodes: [max_pool2d], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   max_pool2d => getitem
# Graph fragment:
#   %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 0), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x4 = xindex
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 5, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = (-1) + x1
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = (-1) + x0
    tmp15 = tmp14 < tmp12
    tmp16 = tmp13 & tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + ((-5) + x4), tmp17 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, float("-inf"), tmp18.dtype)
    tmp20 = tl.where(tmp10, tmp18, tmp19)
    tmp21 = x0
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = (-1) + x1
    tmp27 = tl.full([1], 4, tl.int64)
    tmp28 = tmp26 < tmp27
    tmp29 = x0
    tmp30 = tmp29 < tmp27
    tmp31 = tmp28 & tmp30
    tmp32 = tmp31 & tmp25
    tmp33 = tl.load(in_ptr0 + ((-4) + x4), tmp32 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, float("-inf"), tmp33.dtype)
    tmp35 = tl.where(tmp25, tmp33, tmp34)
    tmp36 = triton_helpers.maximum(tmp35, tmp20)
    tmp37 = 1 + x0
    tmp38 = tmp37 >= tmp1
    tmp39 = tmp37 < tmp3
    tmp40 = tmp38 & tmp39
    tmp41 = tmp5 & tmp40
    tmp42 = (-1) + x1
    tmp43 = tl.full([1], 4, tl.int64)
    tmp44 = tmp42 < tmp43
    tmp45 = 1 + x0
    tmp46 = tmp45 < tmp43
    tmp47 = tmp44 & tmp46
    tmp48 = tmp47 & tmp41
    tmp49 = tl.load(in_ptr0 + ((-3) + x4), tmp48 & xmask, other=0.0)
    tmp50 = tl.full(tmp49.shape, float("-inf"), tmp49.dtype)
    tmp51 = tl.where(tmp41, tmp49, tmp50)
    tmp52 = triton_helpers.maximum(tmp51, tmp36)
    tmp53 = 2 + x0
    tmp54 = tmp53 >= tmp1
    tmp55 = tmp53 < tmp3
    tmp56 = tmp54 & tmp55
    tmp57 = tmp5 & tmp56
    tmp58 = (-1) + x1
    tmp59 = tl.full([1], 4, tl.int64)
    tmp60 = tmp58 < tmp59
    tmp61 = 2 + x0
    tmp62 = tmp61 < tmp59
    tmp63 = tmp60 & tmp62
    tmp64 = tmp63 & tmp57
    tmp65 = tl.load(in_ptr0 + ((-2) + x4), tmp64 & xmask, other=0.0)
    tmp66 = tl.full(tmp65.shape, float("-inf"), tmp65.dtype)
    tmp67 = tl.where(tmp57, tmp65, tmp66)
    tmp68 = triton_helpers.maximum(tmp67, tmp52)
    tmp69 = x1
    tmp70 = tmp69 >= tmp1
    tmp71 = tmp69 < tmp3
    tmp72 = tmp70 & tmp71
    tmp73 = tmp72 & tmp9
    tmp74 = x1
    tmp75 = tl.full([1], 4, tl.int64)
    tmp76 = tmp74 < tmp75
    tmp77 = (-1) + x0
    tmp78 = tmp77 < tmp75
    tmp79 = tmp76 & tmp78
    tmp80 = tmp79 & tmp73
    tmp81 = tl.load(in_ptr0 + ((-1) + x4), tmp80 & xmask, other=0.0)
    tmp82 = tl.full(tmp81.shape, float("-inf"), tmp81.dtype)
    tmp83 = tl.where(tmp73, tmp81, tmp82)
    tmp84 = triton_helpers.maximum(tmp83, tmp68)
    tmp85 = tmp72 & tmp24
    tmp86 = x1
    tmp87 = tl.full([1], 4, tl.int64)
    tmp88 = tmp86 < tmp87
    tmp89 = x0
    tmp90 = tmp89 < tmp87
    tmp91 = tmp88 & tmp90
    tmp92 = tmp91 & tmp85
    tmp93 = tl.load(in_ptr0 + (x4), tmp92 & xmask, other=0.0)
    tmp94 = tl.full(tmp93.shape, float("-inf"), tmp93.dtype)
    tmp95 = tl.where(tmp85, tmp93, tmp94)
    tmp96 = triton_helpers.maximum(tmp95, tmp84)
    tmp97 = tmp72 & tmp40
    tmp98 = x1
    tmp99 = tl.full([1], 4, tl.int64)
    tmp100 = tmp98 < tmp99
    tmp101 = 1 + x0
    tmp102 = tmp101 < tmp99
    tmp103 = tmp100 & tmp102
    tmp104 = tmp103 & tmp97
    tmp105 = tl.load(in_ptr0 + (1 + x4), tmp104 & xmask, other=0.0)
    tmp106 = tl.full(tmp105.shape, float("-inf"), tmp105.dtype)
    tmp107 = tl.where(tmp97, tmp105, tmp106)
    tmp108 = triton_helpers.maximum(tmp107, tmp96)
    tmp109 = tmp72 & tmp56
    tmp110 = x1
    tmp111 = tl.full([1], 4, tl.int64)
    tmp112 = tmp110 < tmp111
    tmp113 = 2 + x0
    tmp114 = tmp113 < tmp111
    tmp115 = tmp112 & tmp114
    tmp116 = tmp115 & tmp109
    tmp117 = tl.load(in_ptr0 + (2 + x4), tmp116 & xmask, other=0.0)
    tmp118 = tl.full(tmp117.shape, float("-inf"), tmp117.dtype)
    tmp119 = tl.where(tmp109, tmp117, tmp118)
    tmp120 = triton_helpers.maximum(tmp119, tmp108)
    tmp121 = 1 + x1
    tmp122 = tmp121 >= tmp1
    tmp123 = tmp121 < tmp3
    tmp124 = tmp122 & tmp123
    tmp125 = tmp124 & tmp9
    tmp126 = 1 + x1
    tmp127 = tl.full([1], 4, tl.int64)
    tmp128 = tmp126 < tmp127
    tmp129 = (-1) + x0
    tmp130 = tmp129 < tmp127
    tmp131 = tmp128 & tmp130
    tmp132 = tmp131 & tmp125
    tmp133 = tl.load(in_ptr0 + (3 + x4), tmp132 & xmask, other=0.0)
    tmp134 = tl.full(tmp133.shape, float("-inf"), tmp133.dtype)
    tmp135 = tl.where(tmp125, tmp133, tmp134)
    tmp136 = triton_helpers.maximum(tmp135, tmp120)
    tmp137 = tmp124 & tmp24
    tmp138 = 1 + x1
    tmp139 = tl.full([1], 4, tl.int64)
    tmp140 = tmp138 < tmp139
    tmp141 = x0
    tmp142 = tmp141 < tmp139
    tmp143 = tmp140 & tmp142
    tmp144 = tmp143 & tmp137
    tmp145 = tl.load(in_ptr0 + (4 + x4), tmp144 & xmask, other=0.0)
    tmp146 = tl.full(tmp145.shape, float("-inf"), tmp145.dtype)
    tmp147 = tl.where(tmp137, tmp145, tmp146)
    tmp148 = triton_helpers.maximum(tmp147, tmp136)
    tmp149 = tmp124 & tmp40
    tmp150 = 1 + x1
    tmp151 = tl.full([1], 4, tl.int64)
    tmp152 = tmp150 < tmp151
    tmp153 = 1 + x0
    tmp154 = tmp153 < tmp151
    tmp155 = tmp152 & tmp154
    tmp156 = tmp155 & tmp149
    tmp157 = tl.load(in_ptr0 + (5 + x4), tmp156 & xmask, other=0.0)
    tmp158 = tl.full(tmp157.shape, float("-inf"), tmp157.dtype)
    tmp159 = tl.where(tmp149, tmp157, tmp158)
    tmp160 = triton_helpers.maximum(tmp159, tmp148)
    tmp161 = tmp124 & tmp56
    tmp162 = 1 + x1
    tmp163 = tl.full([1], 4, tl.int64)
    tmp164 = tmp162 < tmp163
    tmp165 = 2 + x0
    tmp166 = tmp165 < tmp163
    tmp167 = tmp164 & tmp166
    tmp168 = tmp167 & tmp161
    tmp169 = tl.load(in_ptr0 + (6 + x4), tmp168 & xmask, other=0.0)
    tmp170 = tl.full(tmp169.shape, float("-inf"), tmp169.dtype)
    tmp171 = tl.where(tmp161, tmp169, tmp170)
    tmp172 = triton_helpers.maximum(tmp171, tmp160)
    tmp173 = 2 + x1
    tmp174 = tmp173 >= tmp1
    tmp175 = tmp173 < tmp3
    tmp176 = tmp174 & tmp175
    tmp177 = tmp176 & tmp9
    tmp178 = 2 + x1
    tmp179 = tl.full([1], 4, tl.int64)
    tmp180 = tmp178 < tmp179
    tmp181 = (-1) + x0
    tmp182 = tmp181 < tmp179
    tmp183 = tmp180 & tmp182
    tmp184 = tmp183 & tmp177
    tmp185 = tl.load(in_ptr0 + (7 + x4), tmp184 & xmask, other=0.0)
    tmp186 = tl.full(tmp185.shape, float("-inf"), tmp185.dtype)
    tmp187 = tl.where(tmp177, tmp185, tmp186)
    tmp188 = triton_helpers.maximum(tmp187, tmp172)
    tmp189 = tmp176 & tmp24
    tmp190 = 2 + x1
    tmp191 = tl.full([1], 4, tl.int64)
    tmp192 = tmp190 < tmp191
    tmp193 = x0
    tmp194 = tmp193 < tmp191
    tmp195 = tmp192 & tmp194
    tmp196 = tmp195 & tmp189
    tmp197 = tl.load(in_ptr0 + (8 + x4), tmp196 & xmask, other=0.0)
    tmp198 = tl.full(tmp197.shape, float("-inf"), tmp197.dtype)
    tmp199 = tl.where(tmp189, tmp197, tmp198)
    tmp200 = triton_helpers.maximum(tmp199, tmp188)
    tmp201 = tmp176 & tmp40
    tmp202 = 2 + x1
    tmp203 = tl.full([1], 4, tl.int64)
    tmp204 = tmp202 < tmp203
    tmp205 = 1 + x0
    tmp206 = tmp205 < tmp203
    tmp207 = tmp204 & tmp206
    tmp208 = tmp207 & tmp201
    tmp209 = tl.load(in_ptr0 + (9 + x4), tmp208 & xmask, other=0.0)
    tmp210 = tl.full(tmp209.shape, float("-inf"), tmp209.dtype)
    tmp211 = tl.where(tmp201, tmp209, tmp210)
    tmp212 = triton_helpers.maximum(tmp211, tmp200)
    tmp213 = tmp176 & tmp56
    tmp214 = 2 + x1
    tmp215 = tl.full([1], 4, tl.int64)
    tmp216 = tmp214 < tmp215
    tmp217 = 2 + x0
    tmp218 = tmp217 < tmp215
    tmp219 = tmp216 & tmp218
    tmp220 = tmp219 & tmp213
    tmp221 = tl.load(in_ptr0 + (10 + x4), tmp220 & xmask, other=0.0)
    tmp222 = tl.full(tmp221.shape, float("-inf"), tmp221.dtype)
    tmp223 = tl.where(tmp213, tmp221, tmp222)
    tmp224 = triton_helpers.maximum(tmp223, tmp212)
    tl.store(out_ptr0 + (x4), tmp224, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_pool2d], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
