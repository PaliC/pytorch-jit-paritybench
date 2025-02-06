
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_rsub_sub_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_cat_clamp_floor_mul_rsub_sub_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex // 16
    x7 = xindex
    x5 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr9 + (x1), xmask, eviction_policy='evict_last')
    tmp143 = tl.load(in_ptr10 + (x1), xmask, eviction_policy='evict_last')
    tmp179 = tl.load(in_ptr11 + (x1), xmask, eviction_policy='evict_last')
    tmp271 = tl.load(in_ptr15 + (x7), xmask)
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = x2
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tl.full([1], 2, tl.int64)
    tmp13 = tmp9 < tmp12
    tmp14 = tl.load(in_ptr2 + (tmp8 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp9 >= tmp12
    tmp16 = tl.full([1], 3, tl.int64)
    tmp17 = tmp9 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr3 + (tmp8 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp9 >= tmp16
    tmp21 = tl.full([1], 4, tl.int64)
    tmp22 = tmp9 < tmp21
    tmp23 = tl.load(in_ptr4 + (tmp8 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp24 = tl.where(tmp18, tmp19, tmp23)
    tmp25 = tl.where(tmp13, tmp14, tmp24)
    tmp26 = tl.load(in_ptr5 + (tmp8 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp27 = tmp25 + tmp26
    tmp28 = x0
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 0.3333333333333333
    tmp31 = tmp29 * tmp30
    tmp32 = libdevice.floor(tmp31)
    tmp33 = tmp31 - tmp32
    tmp34 = 0.0
    tmp35 = triton_helpers.maximum(tmp33, tmp34)
    tmp36 = 1.0
    tmp37 = triton_helpers.minimum(tmp35, tmp36)
    tmp38 = tmp37 + tmp36
    tmp39 = -0.75
    tmp40 = tmp38 * tmp39
    tmp41 = -3.75
    tmp42 = tmp40 - tmp41
    tmp43 = tmp42 * tmp38
    tmp44 = -6.0
    tmp45 = tmp43 + tmp44
    tmp46 = tmp45 * tmp38
    tmp47 = -3.0
    tmp48 = tmp46 - tmp47
    tmp49 = tmp27 * tmp48
    tmp51 = tmp50 + tmp1
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr2 + (tmp53 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp55 = tl.load(in_ptr3 + (tmp53 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr4 + (tmp53 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.where(tmp18, tmp55, tmp56)
    tmp58 = tl.where(tmp13, tmp54, tmp57)
    tmp59 = tl.load(in_ptr5 + (tmp53 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp60 = tmp58 + tmp59
    tmp61 = 1.25
    tmp62 = tmp37 * tmp61
    tmp63 = 2.25
    tmp64 = tmp62 - tmp63
    tmp65 = tmp64 * tmp37
    tmp66 = tmp65 * tmp37
    tmp67 = tmp66 + tmp36
    tmp68 = tmp60 * tmp67
    tmp70 = tmp69 + tmp1
    tmp71 = tmp69 < 0
    tmp72 = tl.where(tmp71, tmp70, tmp69)
    tmp73 = tl.load(in_ptr2 + (tmp72 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = tl.load(in_ptr3 + (tmp72 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tl.load(in_ptr4 + (tmp72 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.where(tmp18, tmp74, tmp75)
    tmp77 = tl.where(tmp13, tmp73, tmp76)
    tmp78 = tl.load(in_ptr5 + (tmp72 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp79 = tmp77 + tmp78
    tmp80 = tmp36 - tmp37
    tmp81 = tmp80 * tmp61
    tmp82 = tmp81 - tmp63
    tmp83 = tmp82 * tmp80
    tmp84 = tmp83 * tmp80
    tmp85 = tmp84 + tmp36
    tmp86 = tmp79 * tmp85
    tmp88 = tmp87 + tmp1
    tmp89 = tmp87 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp87)
    tmp91 = tl.load(in_ptr2 + (tmp90 + 2*tmp4 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp92 = tl.load(in_ptr3 + (tmp90 + 2*tmp4 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tl.load(in_ptr4 + (tmp90 + 2*tmp4 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp94 = tl.where(tmp18, tmp92, tmp93)
    tmp95 = tl.where(tmp13, tmp91, tmp94)
    tmp96 = tl.load(in_ptr5 + (tmp90 + 2*tmp4 + 4*x4), xmask, eviction_policy='evict_last')
    tmp97 = tmp95 + tmp96
    tmp98 = 2.0
    tmp99 = tmp98 - tmp37
    tmp100 = tmp99 * tmp39
    tmp101 = tmp100 - tmp41
    tmp102 = tmp101 * tmp99
    tmp103 = tmp102 + tmp44
    tmp104 = tmp103 * tmp99
    tmp105 = tmp104 - tmp47
    tmp106 = tmp97 * tmp105
    tmp108 = tmp107 + tmp1
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tl.load(in_ptr2 + (tmp8 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp112 = tl.load(in_ptr3 + (tmp8 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tl.load(in_ptr4 + (tmp8 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tl.where(tmp18, tmp112, tmp113)
    tmp115 = tl.where(tmp13, tmp111, tmp114)
    tmp116 = tl.load(in_ptr5 + (tmp8 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp117 = tmp115 + tmp116
    tmp118 = tmp117 * tmp48
    tmp119 = tl.load(in_ptr2 + (tmp53 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp120 = tl.load(in_ptr3 + (tmp53 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp121 = tl.load(in_ptr4 + (tmp53 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp122 = tl.where(tmp18, tmp120, tmp121)
    tmp123 = tl.where(tmp13, tmp119, tmp122)
    tmp124 = tl.load(in_ptr5 + (tmp53 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp125 = tmp123 + tmp124
    tmp126 = tmp125 * tmp67
    tmp127 = tl.load(in_ptr2 + (tmp72 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp128 = tl.load(in_ptr3 + (tmp72 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp129 = tl.load(in_ptr4 + (tmp72 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp130 = tl.where(tmp18, tmp128, tmp129)
    tmp131 = tl.where(tmp13, tmp127, tmp130)
    tmp132 = tl.load(in_ptr5 + (tmp72 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp133 = tmp131 + tmp132
    tmp134 = tmp133 * tmp85
    tmp135 = tl.load(in_ptr2 + (tmp90 + 2*tmp110 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr3 + (tmp90 + 2*tmp110 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp137 = tl.load(in_ptr4 + (tmp90 + 2*tmp110 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp138 = tl.where(tmp18, tmp136, tmp137)
    tmp139 = tl.where(tmp13, tmp135, tmp138)
    tmp140 = tl.load(in_ptr5 + (tmp90 + 2*tmp110 + 4*x4), xmask, eviction_policy='evict_last')
    tmp141 = tmp139 + tmp140
    tmp142 = tmp141 * tmp105
    tmp144 = tmp143 + tmp1
    tmp145 = tmp143 < 0
    tmp146 = tl.where(tmp145, tmp144, tmp143)
    tmp147 = tl.load(in_ptr2 + (tmp8 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp148 = tl.load(in_ptr3 + (tmp8 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp149 = tl.load(in_ptr4 + (tmp8 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp150 = tl.where(tmp18, tmp148, tmp149)
    tmp151 = tl.where(tmp13, tmp147, tmp150)
    tmp152 = tl.load(in_ptr5 + (tmp8 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp153 = tmp151 + tmp152
    tmp154 = tmp153 * tmp48
    tmp155 = tl.load(in_ptr2 + (tmp53 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp156 = tl.load(in_ptr3 + (tmp53 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp157 = tl.load(in_ptr4 + (tmp53 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp158 = tl.where(tmp18, tmp156, tmp157)
    tmp159 = tl.where(tmp13, tmp155, tmp158)
    tmp160 = tl.load(in_ptr5 + (tmp53 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp161 = tmp159 + tmp160
    tmp162 = tmp161 * tmp67
    tmp163 = tl.load(in_ptr2 + (tmp72 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp164 = tl.load(in_ptr3 + (tmp72 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp165 = tl.load(in_ptr4 + (tmp72 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp166 = tl.where(tmp18, tmp164, tmp165)
    tmp167 = tl.where(tmp13, tmp163, tmp166)
    tmp168 = tl.load(in_ptr5 + (tmp72 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp169 = tmp167 + tmp168
    tmp170 = tmp169 * tmp85
    tmp171 = tl.load(in_ptr2 + (tmp90 + 2*tmp146 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp172 = tl.load(in_ptr3 + (tmp90 + 2*tmp146 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp173 = tl.load(in_ptr4 + (tmp90 + 2*tmp146 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp174 = tl.where(tmp18, tmp172, tmp173)
    tmp175 = tl.where(tmp13, tmp171, tmp174)
    tmp176 = tl.load(in_ptr5 + (tmp90 + 2*tmp146 + 4*x4), xmask, eviction_policy='evict_last')
    tmp177 = tmp175 + tmp176
    tmp178 = tmp177 * tmp105
    tmp180 = tmp179 + tmp1
    tmp181 = tmp179 < 0
    tmp182 = tl.where(tmp181, tmp180, tmp179)
    tmp183 = tl.load(in_ptr2 + (tmp8 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp184 = tl.load(in_ptr3 + (tmp8 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp185 = tl.load(in_ptr4 + (tmp8 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp186 = tl.where(tmp18, tmp184, tmp185)
    tmp187 = tl.where(tmp13, tmp183, tmp186)
    tmp188 = tl.load(in_ptr5 + (tmp8 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp189 = tmp187 + tmp188
    tmp190 = tmp189 * tmp48
    tmp191 = tl.load(in_ptr2 + (tmp53 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp192 = tl.load(in_ptr3 + (tmp53 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp193 = tl.load(in_ptr4 + (tmp53 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp194 = tl.where(tmp18, tmp192, tmp193)
    tmp195 = tl.where(tmp13, tmp191, tmp194)
    tmp196 = tl.load(in_ptr5 + (tmp53 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp197 = tmp195 + tmp196
    tmp198 = tmp197 * tmp67
    tmp199 = tl.load(in_ptr2 + (tmp72 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp200 = tl.load(in_ptr3 + (tmp72 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp201 = tl.load(in_ptr4 + (tmp72 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp202 = tl.where(tmp18, tmp200, tmp201)
    tmp203 = tl.where(tmp13, tmp199, tmp202)
    tmp204 = tl.load(in_ptr5 + (tmp72 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp205 = tmp203 + tmp204
    tmp206 = tmp205 * tmp85
    tmp207 = tl.load(in_ptr2 + (tmp90 + 2*tmp182 + 4*(x2) + 8*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp208 = tl.load(in_ptr3 + (tmp90 + 2*tmp182 + 4*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp209 = tl.load(in_ptr4 + (tmp90 + 2*tmp182 + 4*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp210 = tl.where(tmp18, tmp208, tmp209)
    tmp211 = tl.where(tmp13, tmp207, tmp210)
    tmp212 = tl.load(in_ptr5 + (tmp90 + 2*tmp182 + 4*x4), xmask, eviction_policy='evict_last')
    tmp213 = tmp211 + tmp212
    tmp214 = tmp213 * tmp105
    tmp215 = tmp49 + tmp68
    tmp216 = tmp215 + tmp86
    tmp217 = tmp216 + tmp106
    tmp218 = x1
    tmp219 = tmp218.to(tl.float32)
    tmp220 = tmp219 * tmp30
    tmp221 = libdevice.floor(tmp220)
    tmp222 = tmp220 - tmp221
    tmp223 = triton_helpers.maximum(tmp222, tmp34)
    tmp224 = triton_helpers.minimum(tmp223, tmp36)
    tmp225 = tmp224 + tmp36
    tmp226 = tmp225 * tmp39
    tmp227 = tmp226 - tmp41
    tmp228 = tmp227 * tmp225
    tmp229 = tmp228 + tmp44
    tmp230 = tmp229 * tmp225
    tmp231 = tmp230 - tmp47
    tmp232 = tmp217 * tmp231
    tmp233 = tmp118 + tmp126
    tmp234 = tmp233 + tmp134
    tmp235 = tmp234 + tmp142
    tmp236 = tmp224 * tmp61
    tmp237 = tmp236 - tmp63
    tmp238 = tmp237 * tmp224
    tmp239 = tmp238 * tmp224
    tmp240 = tmp239 + tmp36
    tmp241 = tmp235 * tmp240
    tmp242 = tmp232 + tmp241
    tmp243 = tmp154 + tmp162
    tmp244 = tmp243 + tmp170
    tmp245 = tmp244 + tmp178
    tmp246 = tmp36 - tmp224
    tmp247 = tmp246 * tmp61
    tmp248 = tmp247 - tmp63
    tmp249 = tmp248 * tmp246
    tmp250 = tmp249 * tmp246
    tmp251 = tmp250 + tmp36
    tmp252 = tmp245 * tmp251
    tmp253 = tmp242 + tmp252
    tmp254 = tmp190 + tmp198
    tmp255 = tmp254 + tmp206
    tmp256 = tmp255 + tmp214
    tmp257 = tmp98 - tmp224
    tmp258 = tmp257 * tmp39
    tmp259 = tmp258 - tmp41
    tmp260 = tmp259 * tmp257
    tmp261 = tmp260 + tmp44
    tmp262 = tmp261 * tmp257
    tmp263 = tmp262 - tmp47
    tmp264 = tmp256 * tmp263
    tmp265 = tmp253 + tmp264
    tmp266 = tl.load(in_ptr12 + (x5 + 16*(x2) + 32*x3), tmp13 & xmask, other=0.0)
    tmp267 = tl.load(in_ptr13 + (x5 + 16*x3), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp268 = tl.load(in_ptr14 + (x5 + 16*x3), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
    tmp269 = tl.where(tmp18, tmp267, tmp268)
    tmp270 = tl.where(tmp13, tmp266, tmp269)
    tmp272 = tmp270 + tmp271
    tmp273 = tmp272 + tmp265
    tl.store(in_out_ptr0 + (x7), tmp273, xmask)
