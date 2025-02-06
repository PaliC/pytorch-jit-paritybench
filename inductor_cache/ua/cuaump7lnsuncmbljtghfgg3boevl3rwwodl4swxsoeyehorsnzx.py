
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_logsumexp_mean_neg_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 36, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_logsumexp_mean_neg_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp21 = tl.load(in_ptr0 + (1))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (4))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp27 = tl.load(in_ptr1 + (5))
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK])
    tmp31 = tl.load(in_ptr1 + (6))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp35 = tl.load(in_ptr1 + (7))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK])
    tmp42 = tl.load(in_ptr0 + (2))
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp45 = tl.load(in_ptr1 + (8))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp48 = tl.load(in_ptr1 + (9))
    tmp49 = tl.broadcast_to(tmp48, [XBLOCK])
    tmp52 = tl.load(in_ptr1 + (10))
    tmp53 = tl.broadcast_to(tmp52, [XBLOCK])
    tmp56 = tl.load(in_ptr1 + (11))
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK])
    tmp63 = tl.load(in_ptr0 + (3))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp66 = tl.load(in_ptr1 + (12))
    tmp67 = tl.broadcast_to(tmp66, [XBLOCK])
    tmp69 = tl.load(in_ptr1 + (13))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp73 = tl.load(in_ptr1 + (14))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp77 = tl.load(in_ptr1 + (15))
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK])
    tmp86 = tl.load(in_ptr2 + (0))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp88 = tl.load(in_ptr2 + (1))
    tmp89 = tl.broadcast_to(tmp88, [XBLOCK])
    tmp91 = tl.load(in_ptr2 + (2))
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK])
    tmp94 = tl.load(in_ptr2 + (3))
    tmp95 = tl.broadcast_to(tmp94, [XBLOCK])
    tmp115 = tl.load(in_ptr2 + (4))
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK])
    tmp117 = tl.load(in_ptr2 + (5))
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK])
    tmp120 = tl.load(in_ptr2 + (6))
    tmp121 = tl.broadcast_to(tmp120, [XBLOCK])
    tmp123 = tl.load(in_ptr2 + (7))
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK])
    tmp143 = tl.load(in_ptr2 + (8))
    tmp144 = tl.broadcast_to(tmp143, [XBLOCK])
    tmp145 = tl.load(in_ptr2 + (9))
    tmp146 = tl.broadcast_to(tmp145, [XBLOCK])
    tmp148 = tl.load(in_ptr2 + (10))
    tmp149 = tl.broadcast_to(tmp148, [XBLOCK])
    tmp151 = tl.load(in_ptr2 + (11))
    tmp152 = tl.broadcast_to(tmp151, [XBLOCK])
    tmp171 = tl.load(in_ptr2 + (12))
    tmp172 = tl.broadcast_to(tmp171, [XBLOCK])
    tmp173 = tl.load(in_ptr2 + (13))
    tmp174 = tl.broadcast_to(tmp173, [XBLOCK])
    tmp176 = tl.load(in_ptr2 + (14))
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK])
    tmp179 = tl.load(in_ptr2 + (15))
    tmp180 = tl.broadcast_to(tmp179, [XBLOCK])
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp6 = tmp5 * tmp2
    tmp9 = tmp8 * tmp2
    tmp10 = triton_helpers.maximum(tmp6, tmp9)
    tmp13 = tmp12 * tmp2
    tmp14 = triton_helpers.maximum(tmp10, tmp13)
    tmp17 = tmp16 * tmp2
    tmp18 = triton_helpers.maximum(tmp14, tmp17)
    tmp19 = tmp3 - tmp18
    tmp20 = -tmp19
    tmp23 = tmp22 * tmp2
    tmp26 = tmp25 * tmp2
    tmp29 = tmp28 * tmp2
    tmp30 = triton_helpers.maximum(tmp26, tmp29)
    tmp33 = tmp32 * tmp2
    tmp34 = triton_helpers.maximum(tmp30, tmp33)
    tmp37 = tmp36 * tmp2
    tmp38 = triton_helpers.maximum(tmp34, tmp37)
    tmp39 = tmp23 - tmp38
    tmp40 = -tmp39
    tmp41 = tmp20 + tmp40
    tmp44 = tmp43 * tmp2
    tmp47 = tmp46 * tmp2
    tmp50 = tmp49 * tmp2
    tmp51 = triton_helpers.maximum(tmp47, tmp50)
    tmp54 = tmp53 * tmp2
    tmp55 = triton_helpers.maximum(tmp51, tmp54)
    tmp58 = tmp57 * tmp2
    tmp59 = triton_helpers.maximum(tmp55, tmp58)
    tmp60 = tmp44 - tmp59
    tmp61 = -tmp60
    tmp62 = tmp41 + tmp61
    tmp65 = tmp64 * tmp2
    tmp68 = tmp67 * tmp2
    tmp71 = tmp70 * tmp2
    tmp72 = triton_helpers.maximum(tmp68, tmp71)
    tmp75 = tmp74 * tmp2
    tmp76 = triton_helpers.maximum(tmp72, tmp75)
    tmp79 = tmp78 * tmp2
    tmp80 = triton_helpers.maximum(tmp76, tmp79)
    tmp81 = tmp65 - tmp80
    tmp82 = -tmp81
    tmp83 = tmp62 + tmp82
    tmp84 = 4.0
    tmp85 = tmp83 / tmp84
    tmp90 = triton_helpers.maximum(tmp87, tmp89)
    tmp93 = triton_helpers.maximum(tmp90, tmp92)
    tmp96 = triton_helpers.maximum(tmp93, tmp95)
    tmp97 = tl_math.abs(tmp96)
    tmp98 = float("inf")
    tmp99 = tmp97 == tmp98
    tmp100 = 0.0
    tmp101 = tl.where(tmp99, tmp100, tmp96)
    tmp102 = tmp87 - tmp101
    tmp103 = tl_math.exp(tmp102)
    tmp104 = tmp89 - tmp101
    tmp105 = tl_math.exp(tmp104)
    tmp106 = tmp103 + tmp105
    tmp107 = tmp92 - tmp101
    tmp108 = tl_math.exp(tmp107)
    tmp109 = tmp106 + tmp108
    tmp110 = tmp95 - tmp101
    tmp111 = tl_math.exp(tmp110)
    tmp112 = tmp109 + tmp111
    tmp113 = tl_math.log(tmp112)
    tmp114 = tmp113 + tmp101
    tmp119 = triton_helpers.maximum(tmp116, tmp118)
    tmp122 = triton_helpers.maximum(tmp119, tmp121)
    tmp125 = triton_helpers.maximum(tmp122, tmp124)
    tmp126 = tl_math.abs(tmp125)
    tmp127 = tmp126 == tmp98
    tmp128 = tl.where(tmp127, tmp100, tmp125)
    tmp129 = tmp116 - tmp128
    tmp130 = tl_math.exp(tmp129)
    tmp131 = tmp118 - tmp128
    tmp132 = tl_math.exp(tmp131)
    tmp133 = tmp130 + tmp132
    tmp134 = tmp121 - tmp128
    tmp135 = tl_math.exp(tmp134)
    tmp136 = tmp133 + tmp135
    tmp137 = tmp124 - tmp128
    tmp138 = tl_math.exp(tmp137)
    tmp139 = tmp136 + tmp138
    tmp140 = tl_math.log(tmp139)
    tmp141 = tmp140 + tmp128
    tmp142 = tmp114 + tmp141
    tmp147 = triton_helpers.maximum(tmp144, tmp146)
    tmp150 = triton_helpers.maximum(tmp147, tmp149)
    tmp153 = triton_helpers.maximum(tmp150, tmp152)
    tmp154 = tl_math.abs(tmp153)
    tmp155 = tmp154 == tmp98
    tmp156 = tl.where(tmp155, tmp100, tmp153)
    tmp157 = tmp144 - tmp156
    tmp158 = tl_math.exp(tmp157)
    tmp159 = tmp146 - tmp156
    tmp160 = tl_math.exp(tmp159)
    tmp161 = tmp158 + tmp160
    tmp162 = tmp149 - tmp156
    tmp163 = tl_math.exp(tmp162)
    tmp164 = tmp161 + tmp163
    tmp165 = tmp152 - tmp156
    tmp166 = tl_math.exp(tmp165)
    tmp167 = tmp164 + tmp166
    tmp168 = tl_math.log(tmp167)
    tmp169 = tmp168 + tmp156
    tmp170 = tmp142 + tmp169
    tmp175 = triton_helpers.maximum(tmp172, tmp174)
    tmp178 = triton_helpers.maximum(tmp175, tmp177)
    tmp181 = triton_helpers.maximum(tmp178, tmp180)
    tmp182 = tl_math.abs(tmp181)
    tmp183 = tmp182 == tmp98
    tmp184 = tl.where(tmp183, tmp100, tmp181)
    tmp185 = tmp172 - tmp184
    tmp186 = tl_math.exp(tmp185)
    tmp187 = tmp174 - tmp184
    tmp188 = tl_math.exp(tmp187)
    tmp189 = tmp186 + tmp188
    tmp190 = tmp177 - tmp184
    tmp191 = tl_math.exp(tmp190)
    tmp192 = tmp189 + tmp191
    tmp193 = tmp180 - tmp184
    tmp194 = tl_math.exp(tmp193)
    tmp195 = tmp192 + tmp194
    tmp196 = tl_math.log(tmp195)
    tmp197 = tmp196 + tmp184
    tmp198 = tmp170 + tmp197
    tmp199 = tmp198 / tmp84
    tmp200 = tmp85 + tmp199
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp85, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp199, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp200, None)
