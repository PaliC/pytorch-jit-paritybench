# AOT ID: ['50_inference']
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


# kernel path: inductor_cache/is/cisvofwnbheoik5dwyaxrlwkyqcwcfxrblbwyzbk32mcjuyqfajp.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%arg0_1, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_0 = async_compile.triton('triton_poi_fused_avg_pool2d_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x3 = xindex // 2
    x4 = xindex
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x3), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + 2*x0 + 8*x3), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x3), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 8*x3), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x3), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + 2*x0 + 8*x3), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x3), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x3), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = 1 + ((-2)*x0) + ((-2)*x1) + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5))) + ((-2)*x0*((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))) + ((-2)*x1*((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5)))) + 4*x0*x1 + ((5) * ((5) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (5))) + ((5) * ((5) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (5)))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x4), tmp53, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mx/cmxbbxkdjzx6glh47q35dj252paodqq3vrwjvvdr5vpqczlsuijt.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   x_1 => avg_pool2d_1
#   x_2 => avg_pool2d_2
#   x_3 => avg_pool2d_3
# Graph fragment:
#   %avg_pool2d_1 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d, [3, 3], [2, 2], [1, 1]), kwargs = {})
#   %avg_pool2d_2 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d_1, [3, 3], [2, 2], [1, 1]), kwargs = {})
#   %avg_pool2d_3 : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%avg_pool2d_2, [3, 3], [2, 2], [1, 1]), kwargs = {})
triton_poi_fused_avg_pool2d_1 = async_compile.triton('triton_poi_fused_avg_pool2d_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], -1, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5 & tmp5
    tmp7 = tl.load(in_ptr0 + ((-3) + 4*x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp1 >= tmp1
    tmp9 = tmp1 < tmp3
    tmp10 = tmp8 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-2) + 4*x0), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp12 + tmp7
    tmp14 = tl.full([1], 1, tl.int64)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-1) + 4*x0), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 + tmp13
    tmp21 = tmp10 & tmp5
    tmp22 = tl.load(in_ptr0 + ((-1) + 4*x0), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp22 + tmp20
    tmp24 = tmp10 & tmp10
    tmp25 = tl.load(in_ptr0 + (4*x0), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 + tmp23
    tmp27 = tmp10 & tmp17
    tmp28 = tl.load(in_ptr0 + (1 + 4*x0), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp28 + tmp26
    tmp30 = tmp17 & tmp5
    tmp31 = tl.load(in_ptr0 + (1 + 4*x0), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp29
    tmp33 = tmp17 & tmp10
    tmp34 = tl.load(in_ptr0 + (2 + 4*x0), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp17 & tmp17
    tmp37 = tl.load(in_ptr0 + (3 + 4*x0), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = tl.full([1], 9, tl.int32)
    tmp40 = tmp38 / tmp39
    tmp41 = tmp0 < tmp14
    tmp42 = tmp2 & tmp41
    tmp43 = tmp42 & tmp42
    tmp44 = tl.full([1], -3, tl.int64)
    tmp45 = tl.full([1], 0, tl.int64)
    tmp46 = tmp44 >= tmp45
    tmp47 = tl.full([1], 1, tl.int64)
    tmp48 = tmp44 < tmp47
    tmp49 = tmp46 & tmp48
    tmp50 = tmp49 & tmp49
    tmp51 = tmp50 & tmp43
    tmp52 = tl.full([1], -2, tl.int64)
    tmp53 = tmp52 >= tmp45
    tmp54 = tmp52 < tmp47
    tmp55 = tmp53 & tmp54
    tmp56 = tmp49 & tmp55
    tmp57 = tmp56 & tmp43
    tmp58 = tmp40 + tmp40
    tmp59 = tl.full([1], -1, tl.int64)
    tmp60 = tmp59 >= tmp45
    tmp61 = tmp59 < tmp47
    tmp62 = tmp60 & tmp61
    tmp63 = tmp49 & tmp62
    tmp64 = tmp63 & tmp43
    tmp65 = tmp40 + tmp58
    tmp66 = tmp55 & tmp49
    tmp67 = tmp66 & tmp43
    tmp68 = tmp40 + tmp65
    tmp69 = tmp55 & tmp55
    tmp70 = tmp69 & tmp43
    tmp71 = tmp40 + tmp68
    tmp72 = tmp55 & tmp62
    tmp73 = tmp72 & tmp43
    tmp74 = tmp40 + tmp71
    tmp75 = tmp62 & tmp49
    tmp76 = tmp75 & tmp43
    tmp77 = tmp40 + tmp74
    tmp78 = tmp62 & tmp55
    tmp79 = tmp78 & tmp43
    tmp80 = tmp40 + tmp77
    tmp81 = tmp62 & tmp62
    tmp82 = tmp81 & tmp43
    tmp83 = tmp40 + tmp80
    tmp84 = tl.full([1], 9, tl.int32)
    tmp85 = tmp83 / tmp84
    tmp86 = tl.full(tmp85.shape, 0.0, tmp85.dtype)
    tmp87 = tl.where(tmp43, tmp85, tmp86)
    tmp88 = tmp1 < tmp14
    tmp89 = tmp8 & tmp88
    tmp90 = tmp42 & tmp89
    tmp91 = tl.full([1], -3, tl.int64)
    tmp92 = tl.full([1], 0, tl.int64)
    tmp93 = tmp91 >= tmp92
    tmp94 = tl.full([1], 1, tl.int64)
    tmp95 = tmp91 < tmp94
    tmp96 = tmp93 & tmp95
    tmp97 = tl.full([1], -1, tl.int64)
    tmp98 = tmp97 >= tmp92
    tmp99 = tmp97 < tmp94
    tmp100 = tmp98 & tmp99
    tmp101 = tmp96 & tmp100
    tmp102 = tmp101 & tmp90
    tmp103 = tmp92 >= tmp92
    tmp104 = tmp92 < tmp94
    tmp105 = tmp103 & tmp104
    tmp106 = tmp96 & tmp105
    tmp107 = tmp106 & tmp90
    tmp108 = tmp40 + tmp40
    tmp109 = tmp94 >= tmp92
    tmp110 = tmp94 < tmp94
    tmp111 = tmp109 & tmp110
    tmp112 = tmp96 & tmp111
    tmp113 = tmp112 & tmp90
    tmp114 = tmp40 + tmp108
    tmp115 = tl.full([1], -2, tl.int64)
    tmp116 = tmp115 >= tmp92
    tmp117 = tmp115 < tmp94
    tmp118 = tmp116 & tmp117
    tmp119 = tmp118 & tmp100
    tmp120 = tmp119 & tmp90
    tmp121 = tmp40 + tmp114
    tmp122 = tmp118 & tmp105
    tmp123 = tmp122 & tmp90
    tmp124 = tmp40 + tmp121
    tmp125 = tmp118 & tmp111
    tmp126 = tmp125 & tmp90
    tmp127 = tmp40 + tmp124
    tmp128 = tmp100 & tmp100
    tmp129 = tmp128 & tmp90
    tmp130 = tmp40 + tmp127
    tmp131 = tmp100 & tmp105
    tmp132 = tmp131 & tmp90
    tmp133 = tmp40 + tmp130
    tmp134 = tmp100 & tmp111
    tmp135 = tmp134 & tmp90
    tmp136 = tmp40 + tmp133
    tmp137 = tl.full([1], 9, tl.int32)
    tmp138 = tmp136 / tmp137
    tmp139 = tl.full(tmp138.shape, 0.0, tmp138.dtype)
    tmp140 = tl.where(tmp90, tmp138, tmp139)
    tmp141 = tmp140 + tmp87
    tmp142 = tmp14 < tmp14
    tmp143 = tmp15 & tmp142
    tmp144 = tmp42 & tmp143
    tmp145 = tl.full([1], -3, tl.int64)
    tmp146 = tl.full([1], 0, tl.int64)
    tmp147 = tmp145 >= tmp146
    tmp148 = tl.full([1], 1, tl.int64)
    tmp149 = tmp145 < tmp148
    tmp150 = tmp147 & tmp149
    tmp151 = tmp148 >= tmp146
    tmp152 = tmp148 < tmp148
    tmp153 = tmp151 & tmp152
    tmp154 = tmp150 & tmp153
    tmp155 = tmp154 & tmp144
    tmp156 = tl.full([1], 2, tl.int64)
    tmp157 = tmp156 >= tmp146
    tmp158 = tmp156 < tmp148
    tmp159 = tmp157 & tmp158
    tmp160 = tmp150 & tmp159
    tmp161 = tmp160 & tmp144
    tmp162 = tmp40 + tmp40
    tmp163 = tl.full([1], 3, tl.int64)
    tmp164 = tmp163 >= tmp146
    tmp165 = tmp163 < tmp148
    tmp166 = tmp164 & tmp165
    tmp167 = tmp150 & tmp166
    tmp168 = tmp167 & tmp144
    tmp169 = tmp40 + tmp162
    tmp170 = tl.full([1], -2, tl.int64)
    tmp171 = tmp170 >= tmp146
    tmp172 = tmp170 < tmp148
    tmp173 = tmp171 & tmp172
    tmp174 = tmp173 & tmp153
    tmp175 = tmp174 & tmp144
    tmp176 = tmp40 + tmp169
    tmp177 = tmp173 & tmp159
    tmp178 = tmp177 & tmp144
    tmp179 = tmp40 + tmp176
    tmp180 = tmp173 & tmp166
    tmp181 = tmp180 & tmp144
    tmp182 = tmp40 + tmp179
    tmp183 = tl.full([1], -1, tl.int64)
    tmp184 = tmp183 >= tmp146
    tmp185 = tmp183 < tmp148
    tmp186 = tmp184 & tmp185
    tmp187 = tmp186 & tmp153
    tmp188 = tmp187 & tmp144
    tmp189 = tmp40 + tmp182
    tmp190 = tmp186 & tmp159
    tmp191 = tmp190 & tmp144
    tmp192 = tmp40 + tmp189
    tmp193 = tmp186 & tmp166
    tmp194 = tmp193 & tmp144
    tmp195 = tmp40 + tmp192
    tmp196 = tl.full([1], 3, tl.int32)
    tmp197 = tmp195 / tmp196
    tmp198 = tl.full(tmp197.shape, 0.0, tmp197.dtype)
    tmp199 = tl.where(tmp144, tmp197, tmp198)
    tmp200 = tmp199 + tmp141
    tmp201 = tmp89 & tmp42
    tmp202 = tl.full([1], -1, tl.int64)
    tmp203 = tl.full([1], 0, tl.int64)
    tmp204 = tmp202 >= tmp203
    tmp205 = tl.full([1], 1, tl.int64)
    tmp206 = tmp202 < tmp205
    tmp207 = tmp204 & tmp206
    tmp208 = tl.full([1], -3, tl.int64)
    tmp209 = tmp208 >= tmp203
    tmp210 = tmp208 < tmp205
    tmp211 = tmp209 & tmp210
    tmp212 = tmp207 & tmp211
    tmp213 = tmp212 & tmp201
    tmp214 = tl.full([1], -2, tl.int64)
    tmp215 = tmp214 >= tmp203
    tmp216 = tmp214 < tmp205
    tmp217 = tmp215 & tmp216
    tmp218 = tmp207 & tmp217
    tmp219 = tmp218 & tmp201
    tmp220 = tmp40 + tmp40
    tmp221 = tmp207 & tmp207
    tmp222 = tmp221 & tmp201
    tmp223 = tmp40 + tmp220
    tmp224 = tmp203 >= tmp203
    tmp225 = tmp203 < tmp205
    tmp226 = tmp224 & tmp225
    tmp227 = tmp226 & tmp211
    tmp228 = tmp227 & tmp201
    tmp229 = tmp40 + tmp223
    tmp230 = tmp226 & tmp217
    tmp231 = tmp230 & tmp201
    tmp232 = tmp40 + tmp229
    tmp233 = tmp226 & tmp207
    tmp234 = tmp233 & tmp201
    tmp235 = tmp40 + tmp232
    tmp236 = tmp205 >= tmp203
    tmp237 = tmp205 < tmp205
    tmp238 = tmp236 & tmp237
    tmp239 = tmp238 & tmp211
    tmp240 = tmp239 & tmp201
    tmp241 = tmp40 + tmp235
    tmp242 = tmp238 & tmp217
    tmp243 = tmp242 & tmp201
    tmp244 = tmp40 + tmp241
    tmp245 = tmp238 & tmp207
    tmp246 = tmp245 & tmp201
    tmp247 = tmp40 + tmp244
    tmp248 = tl.full([1], 9, tl.int32)
    tmp249 = tmp247 / tmp248
    tmp250 = tl.full(tmp249.shape, 0.0, tmp249.dtype)
    tmp251 = tl.where(tmp201, tmp249, tmp250)
    tmp252 = tmp251 + tmp200
    tmp253 = tmp89 & tmp89
    tmp254 = tl.full([1], -1, tl.int64)
    tmp255 = tl.full([1], 0, tl.int64)
    tmp256 = tmp254 >= tmp255
    tmp257 = tl.full([1], 1, tl.int64)
    tmp258 = tmp254 < tmp257
    tmp259 = tmp256 & tmp258
    tmp260 = tmp259 & tmp259
    tmp261 = tmp260 & tmp253
    tmp262 = tmp255 >= tmp255
    tmp263 = tmp255 < tmp257
    tmp264 = tmp262 & tmp263
    tmp265 = tmp259 & tmp264
    tmp266 = tmp265 & tmp253
    tmp267 = tmp40 + tmp40
    tmp268 = tmp257 >= tmp255
    tmp269 = tmp257 < tmp257
    tmp270 = tmp268 & tmp269
    tmp271 = tmp259 & tmp270
    tmp272 = tmp271 & tmp253
    tmp273 = tmp40 + tmp267
    tmp274 = tmp264 & tmp259
    tmp275 = tmp274 & tmp253
    tmp276 = tmp40 + tmp273
    tmp277 = tmp264 & tmp264
    tmp278 = tmp277 & tmp253
    tmp279 = tmp40 + tmp276
    tmp280 = tmp264 & tmp270
    tmp281 = tmp280 & tmp253
    tmp282 = tmp40 + tmp279
    tmp283 = tmp270 & tmp259
    tmp284 = tmp283 & tmp253
    tmp285 = tmp40 + tmp282
    tmp286 = tmp270 & tmp264
    tmp287 = tmp286 & tmp253
    tmp288 = tmp40 + tmp285
    tmp289 = tmp270 & tmp270
    tmp290 = tmp289 & tmp253
    tmp291 = tmp40 + tmp288
    tmp292 = tl.full([1], 9, tl.int32)
    tmp293 = tmp291 / tmp292
    tmp294 = tl.full(tmp293.shape, 0.0, tmp293.dtype)
    tmp295 = tl.where(tmp253, tmp293, tmp294)
    tmp296 = tmp295 + tmp252
    tmp297 = tmp89 & tmp143
    tmp298 = tl.full([1], -1, tl.int64)
    tmp299 = tl.full([1], 0, tl.int64)
    tmp300 = tmp298 >= tmp299
    tmp301 = tl.full([1], 1, tl.int64)
    tmp302 = tmp298 < tmp301
    tmp303 = tmp300 & tmp302
    tmp304 = tmp301 >= tmp299
    tmp305 = tmp301 < tmp301
    tmp306 = tmp304 & tmp305
    tmp307 = tmp303 & tmp306
    tmp308 = tmp307 & tmp297
    tmp309 = tl.full([1], 2, tl.int64)
    tmp310 = tmp309 >= tmp299
    tmp311 = tmp309 < tmp301
    tmp312 = tmp310 & tmp311
    tmp313 = tmp303 & tmp312
    tmp314 = tmp313 & tmp297
    tmp315 = tmp40 + tmp40
    tmp316 = tl.full([1], 3, tl.int64)
    tmp317 = tmp316 >= tmp299
    tmp318 = tmp316 < tmp301
    tmp319 = tmp317 & tmp318
    tmp320 = tmp303 & tmp319
    tmp321 = tmp320 & tmp297
    tmp322 = tmp40 + tmp315
    tmp323 = tmp299 >= tmp299
    tmp324 = tmp299 < tmp301
    tmp325 = tmp323 & tmp324
    tmp326 = tmp325 & tmp306
    tmp327 = tmp326 & tmp297
    tmp328 = tmp40 + tmp322
    tmp329 = tmp325 & tmp312
    tmp330 = tmp329 & tmp297
    tmp331 = tmp40 + tmp328
    tmp332 = tmp325 & tmp319
    tmp333 = tmp332 & tmp297
    tmp334 = tmp40 + tmp331
    tmp335 = tmp306 & tmp306
    tmp336 = tmp335 & tmp297
    tmp337 = tmp40 + tmp334
    tmp338 = tmp306 & tmp312
    tmp339 = tmp338 & tmp297
    tmp340 = tmp40 + tmp337
    tmp341 = tmp306 & tmp319
    tmp342 = tmp341 & tmp297
    tmp343 = tmp40 + tmp340
    tmp344 = tl.full([1], 3, tl.int32)
    tmp345 = tmp343 / tmp344
    tmp346 = tl.full(tmp345.shape, 0.0, tmp345.dtype)
    tmp347 = tl.where(tmp297, tmp345, tmp346)
    tmp348 = tmp347 + tmp296
    tmp349 = tmp143 & tmp42
    tmp350 = tl.full([1], 1, tl.int64)
    tmp351 = tl.full([1], 0, tl.int64)
    tmp352 = tmp350 >= tmp351
    tmp353 = tmp350 < tmp350
    tmp354 = tmp352 & tmp353
    tmp355 = tl.full([1], -3, tl.int64)
    tmp356 = tmp355 >= tmp351
    tmp357 = tmp355 < tmp350
    tmp358 = tmp356 & tmp357
    tmp359 = tmp354 & tmp358
    tmp360 = tmp359 & tmp349
    tmp361 = tl.full([1], -2, tl.int64)
    tmp362 = tmp361 >= tmp351
    tmp363 = tmp361 < tmp350
    tmp364 = tmp362 & tmp363
    tmp365 = tmp354 & tmp364
    tmp366 = tmp365 & tmp349
    tmp367 = tmp40 + tmp40
    tmp368 = tl.full([1], -1, tl.int64)
    tmp369 = tmp368 >= tmp351
    tmp370 = tmp368 < tmp350
    tmp371 = tmp369 & tmp370
    tmp372 = tmp354 & tmp371
    tmp373 = tmp372 & tmp349
    tmp374 = tmp40 + tmp367
    tmp375 = tl.full([1], 2, tl.int64)
    tmp376 = tmp375 >= tmp351
    tmp377 = tmp375 < tmp350
    tmp378 = tmp376 & tmp377
    tmp379 = tmp378 & tmp358
    tmp380 = tmp379 & tmp349
    tmp381 = tmp40 + tmp374
    tmp382 = tmp378 & tmp364
    tmp383 = tmp382 & tmp349
    tmp384 = tmp40 + tmp381
    tmp385 = tmp378 & tmp371
    tmp386 = tmp385 & tmp349
    tmp387 = tmp40 + tmp384
    tmp388 = tl.full([1], 3, tl.int64)
    tmp389 = tmp388 >= tmp351
    tmp390 = tmp388 < tmp350
    tmp391 = tmp389 & tmp390
    tmp392 = tmp391 & tmp358
    tmp393 = tmp392 & tmp349
    tmp394 = tmp40 + tmp387
    tmp395 = tmp391 & tmp364
    tmp396 = tmp395 & tmp349
    tmp397 = tmp40 + tmp394
    tmp398 = tmp391 & tmp371
    tmp399 = tmp398 & tmp349
    tmp400 = tmp40 + tmp397
    tmp401 = tl.full([1], 3, tl.int32)
    tmp402 = tmp400 / tmp401
    tmp403 = tl.full(tmp402.shape, 0.0, tmp402.dtype)
    tmp404 = tl.where(tmp349, tmp402, tmp403)
    tmp405 = tmp404 + tmp348
    tmp406 = tmp143 & tmp89
    tmp407 = tl.full([1], 1, tl.int64)
    tmp408 = tl.full([1], 0, tl.int64)
    tmp409 = tmp407 >= tmp408
    tmp410 = tmp407 < tmp407
    tmp411 = tmp409 & tmp410
    tmp412 = tl.full([1], -1, tl.int64)
    tmp413 = tmp412 >= tmp408
    tmp414 = tmp412 < tmp407
    tmp415 = tmp413 & tmp414
    tmp416 = tmp411 & tmp415
    tmp417 = tmp416 & tmp406
    tmp418 = tmp408 >= tmp408
    tmp419 = tmp408 < tmp407
    tmp420 = tmp418 & tmp419
    tmp421 = tmp411 & tmp420
    tmp422 = tmp421 & tmp406
    tmp423 = tmp40 + tmp40
    tmp424 = tmp411 & tmp411
    tmp425 = tmp424 & tmp406
    tmp426 = tmp40 + tmp423
    tmp427 = tl.full([1], 2, tl.int64)
    tmp428 = tmp427 >= tmp408
    tmp429 = tmp427 < tmp407
    tmp430 = tmp428 & tmp429
    tmp431 = tmp430 & tmp415
    tmp432 = tmp431 & tmp406
    tmp433 = tmp40 + tmp426
    tmp434 = tmp430 & tmp420
    tmp435 = tmp434 & tmp406
    tmp436 = tmp40 + tmp433
    tmp437 = tmp430 & tmp411
    tmp438 = tmp437 & tmp406
    tmp439 = tmp40 + tmp436
    tmp440 = tl.full([1], 3, tl.int64)
    tmp441 = tmp440 >= tmp408
    tmp442 = tmp440 < tmp407
    tmp443 = tmp441 & tmp442
    tmp444 = tmp443 & tmp415
    tmp445 = tmp444 & tmp406
    tmp446 = tmp40 + tmp439
    tmp447 = tmp443 & tmp420
    tmp448 = tmp447 & tmp406
    tmp449 = tmp40 + tmp446
    tmp450 = tmp443 & tmp411
    tmp451 = tmp450 & tmp406
    tmp452 = tmp40 + tmp449
    tmp453 = tl.full([1], 3, tl.int32)
    tmp454 = tmp452 / tmp453
    tmp455 = tl.full(tmp454.shape, 0.0, tmp454.dtype)
    tmp456 = tl.where(tmp406, tmp454, tmp455)
    tmp457 = tmp456 + tmp405
    tmp458 = tmp143 & tmp143
    tmp459 = tl.full([1], 1, tl.int64)
    tmp460 = tl.full([1], 0, tl.int64)
    tmp461 = tmp459 >= tmp460
    tmp462 = tmp459 < tmp459
    tmp463 = tmp461 & tmp462
    tmp464 = tmp463 & tmp463
    tmp465 = tmp464 & tmp458
    tmp466 = tl.full([1], 2, tl.int64)
    tmp467 = tmp466 >= tmp460
    tmp468 = tmp466 < tmp459
    tmp469 = tmp467 & tmp468
    tmp470 = tmp463 & tmp469
    tmp471 = tmp470 & tmp458
    tmp472 = tmp40 + tmp40
    tmp473 = tl.full([1], 3, tl.int64)
    tmp474 = tmp473 >= tmp460
    tmp475 = tmp473 < tmp459
    tmp476 = tmp474 & tmp475
    tmp477 = tmp463 & tmp476
    tmp478 = tmp477 & tmp458
    tmp479 = tmp40 + tmp472
    tmp480 = tmp469 & tmp463
    tmp481 = tmp480 & tmp458
    tmp482 = tmp40 + tmp479
    tmp483 = tmp469 & tmp469
    tmp484 = tmp483 & tmp458
    tmp485 = tmp40 + tmp482
    tmp486 = tmp469 & tmp476
    tmp487 = tmp486 & tmp458
    tmp488 = tmp40 + tmp485
    tmp489 = tmp476 & tmp463
    tmp490 = tmp489 & tmp458
    tmp491 = tmp40 + tmp488
    tmp492 = tmp476 & tmp469
    tmp493 = tmp492 & tmp458
    tmp494 = tmp40 + tmp491
    tmp495 = tmp476 & tmp476
    tmp496 = tmp495 & tmp458
    tmp497 = tmp40 + tmp494
    tmp498 = tl.full([1], 1, tl.int32)
    tmp499 = tmp497 / tmp498
    tmp500 = tl.full(tmp499.shape, 0.0, tmp499.dtype)
    tmp501 = tl.where(tmp458, tmp499, tmp500)
    tmp502 = tmp501 + tmp457
    tmp503 = tmp502 / tmp39
    tl.store(in_out_ptr0 + (x0), tmp503, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_0.run(arg0_1, buf0, 64, grid=grid(64), stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf2 = reinterpret_tensor(buf1, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2, x_3], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_1.run(buf2, buf0, 16, grid=grid(16), stream=stream0)
        del buf0
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
