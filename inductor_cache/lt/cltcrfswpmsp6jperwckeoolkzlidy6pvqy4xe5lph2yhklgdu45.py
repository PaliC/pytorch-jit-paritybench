
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_logsumexp_mean_neg_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 36, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_logsumexp_mean_neg_sub_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr1 + (3))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (1))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp23 = tl.load(in_ptr1 + (4))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp26 = tl.load(in_ptr1 + (5))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp30 = tl.load(in_ptr1 + (6))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp34 = tl.load(in_ptr1 + (7))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (2))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp44 = tl.load(in_ptr1 + (8))
    tmp45 = tl.broadcast_to(tmp44, [XBLOCK])
    tmp47 = tl.load(in_ptr1 + (9))
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp51 = tl.load(in_ptr1 + (10))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp55 = tl.load(in_ptr1 + (11))
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK])
    tmp62 = tl.load(in_ptr0 + (3))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp65 = tl.load(in_ptr1 + (12))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp68 = tl.load(in_ptr1 + (13))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
    tmp72 = tl.load(in_ptr1 + (14))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
    tmp76 = tl.load(in_ptr1 + (15))
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
    tmp85 = tl.load(in_ptr2 + (0))
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK])
    tmp87 = tl.load(in_ptr2 + (1))
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK])
    tmp90 = tl.load(in_ptr2 + (2))
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK])
    tmp93 = tl.load(in_ptr2 + (3))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK])
    tmp114 = tl.load(in_ptr2 + (4))
    tmp115 = tl.broadcast_to(tmp114, [XBLOCK])
    tmp116 = tl.load(in_ptr2 + (5))
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK])
    tmp119 = tl.load(in_ptr2 + (6))
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK])
    tmp122 = tl.load(in_ptr2 + (7))
    tmp123 = tl.broadcast_to(tmp122, [XBLOCK])
    tmp142 = tl.load(in_ptr2 + (8))
    tmp143 = tl.broadcast_to(tmp142, [XBLOCK])
    tmp144 = tl.load(in_ptr2 + (9))
    tmp145 = tl.broadcast_to(tmp144, [XBLOCK])
    tmp147 = tl.load(in_ptr2 + (10))
    tmp148 = tl.broadcast_to(tmp147, [XBLOCK])
    tmp150 = tl.load(in_ptr2 + (11))
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK])
    tmp170 = tl.load(in_ptr2 + (12))
    tmp171 = tl.broadcast_to(tmp170, [XBLOCK])
    tmp172 = tl.load(in_ptr2 + (13))
    tmp173 = tl.broadcast_to(tmp172, [XBLOCK])
    tmp175 = tl.load(in_ptr2 + (14))
    tmp176 = tl.broadcast_to(tmp175, [XBLOCK])
    tmp178 = tl.load(in_ptr2 + (15))
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK])
    tmp2 = -tmp1
    tmp5 = -tmp4
    tmp8 = -tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp12 = -tmp11
    tmp13 = triton_helpers.maximum(tmp9, tmp12)
    tmp16 = -tmp15
    tmp17 = triton_helpers.maximum(tmp13, tmp16)
    tmp18 = tmp2 - tmp17
    tmp19 = -tmp18
    tmp22 = -tmp21
    tmp25 = -tmp24
    tmp28 = -tmp27
    tmp29 = triton_helpers.maximum(tmp25, tmp28)
    tmp32 = -tmp31
    tmp33 = triton_helpers.maximum(tmp29, tmp32)
    tmp36 = -tmp35
    tmp37 = triton_helpers.maximum(tmp33, tmp36)
    tmp38 = tmp22 - tmp37
    tmp39 = -tmp38
    tmp40 = tmp19 + tmp39
    tmp43 = -tmp42
    tmp46 = -tmp45
    tmp49 = -tmp48
    tmp50 = triton_helpers.maximum(tmp46, tmp49)
    tmp53 = -tmp52
    tmp54 = triton_helpers.maximum(tmp50, tmp53)
    tmp57 = -tmp56
    tmp58 = triton_helpers.maximum(tmp54, tmp57)
    tmp59 = tmp43 - tmp58
    tmp60 = -tmp59
    tmp61 = tmp40 + tmp60
    tmp64 = -tmp63
    tmp67 = -tmp66
    tmp70 = -tmp69
    tmp71 = triton_helpers.maximum(tmp67, tmp70)
    tmp74 = -tmp73
    tmp75 = triton_helpers.maximum(tmp71, tmp74)
    tmp78 = -tmp77
    tmp79 = triton_helpers.maximum(tmp75, tmp78)
    tmp80 = tmp64 - tmp79
    tmp81 = -tmp80
    tmp82 = tmp61 + tmp81
    tmp83 = 4.0
    tmp84 = tmp82 / tmp83
    tmp89 = triton_helpers.maximum(tmp86, tmp88)
    tmp92 = triton_helpers.maximum(tmp89, tmp91)
    tmp95 = triton_helpers.maximum(tmp92, tmp94)
    tmp96 = tl_math.abs(tmp95)
    tmp97 = float("inf")
    tmp98 = tmp96 == tmp97
    tmp99 = 0.0
    tmp100 = tl.where(tmp98, tmp99, tmp95)
    tmp101 = tmp86 - tmp100
    tmp102 = tl_math.exp(tmp101)
    tmp103 = tmp88 - tmp100
    tmp104 = tl_math.exp(tmp103)
    tmp105 = tmp102 + tmp104
    tmp106 = tmp91 - tmp100
    tmp107 = tl_math.exp(tmp106)
    tmp108 = tmp105 + tmp107
    tmp109 = tmp94 - tmp100
    tmp110 = tl_math.exp(tmp109)
    tmp111 = tmp108 + tmp110
    tmp112 = tl_math.log(tmp111)
    tmp113 = tmp112 + tmp100
    tmp118 = triton_helpers.maximum(tmp115, tmp117)
    tmp121 = triton_helpers.maximum(tmp118, tmp120)
    tmp124 = triton_helpers.maximum(tmp121, tmp123)
    tmp125 = tl_math.abs(tmp124)
    tmp126 = tmp125 == tmp97
    tmp127 = tl.where(tmp126, tmp99, tmp124)
    tmp128 = tmp115 - tmp127
    tmp129 = tl_math.exp(tmp128)
    tmp130 = tmp117 - tmp127
    tmp131 = tl_math.exp(tmp130)
    tmp132 = tmp129 + tmp131
    tmp133 = tmp120 - tmp127
    tmp134 = tl_math.exp(tmp133)
    tmp135 = tmp132 + tmp134
    tmp136 = tmp123 - tmp127
    tmp137 = tl_math.exp(tmp136)
    tmp138 = tmp135 + tmp137
    tmp139 = tl_math.log(tmp138)
    tmp140 = tmp139 + tmp127
    tmp141 = tmp113 + tmp140
    tmp146 = triton_helpers.maximum(tmp143, tmp145)
    tmp149 = triton_helpers.maximum(tmp146, tmp148)
    tmp152 = triton_helpers.maximum(tmp149, tmp151)
    tmp153 = tl_math.abs(tmp152)
    tmp154 = tmp153 == tmp97
    tmp155 = tl.where(tmp154, tmp99, tmp152)
    tmp156 = tmp143 - tmp155
    tmp157 = tl_math.exp(tmp156)
    tmp158 = tmp145 - tmp155
    tmp159 = tl_math.exp(tmp158)
    tmp160 = tmp157 + tmp159
    tmp161 = tmp148 - tmp155
    tmp162 = tl_math.exp(tmp161)
    tmp163 = tmp160 + tmp162
    tmp164 = tmp151 - tmp155
    tmp165 = tl_math.exp(tmp164)
    tmp166 = tmp163 + tmp165
    tmp167 = tl_math.log(tmp166)
    tmp168 = tmp167 + tmp155
    tmp169 = tmp141 + tmp168
    tmp174 = triton_helpers.maximum(tmp171, tmp173)
    tmp177 = triton_helpers.maximum(tmp174, tmp176)
    tmp180 = triton_helpers.maximum(tmp177, tmp179)
    tmp181 = tl_math.abs(tmp180)
    tmp182 = tmp181 == tmp97
    tmp183 = tl.where(tmp182, tmp99, tmp180)
    tmp184 = tmp171 - tmp183
    tmp185 = tl_math.exp(tmp184)
    tmp186 = tmp173 - tmp183
    tmp187 = tl_math.exp(tmp186)
    tmp188 = tmp185 + tmp187
    tmp189 = tmp176 - tmp183
    tmp190 = tl_math.exp(tmp189)
    tmp191 = tmp188 + tmp190
    tmp192 = tmp179 - tmp183
    tmp193 = tl_math.exp(tmp192)
    tmp194 = tmp191 + tmp193
    tmp195 = tl_math.log(tmp194)
    tmp196 = tmp195 + tmp183
    tmp197 = tmp169 + tmp196
    tmp198 = tmp197 / tmp83
    tmp199 = tmp84 + tmp198
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp84, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp198, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK], 0, tl.int32)), tmp199, None)
