
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_avg_pool2d_clamp_div_floor_mul_sign_sub_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_avg_pool2d_clamp_div_floor_mul_sign_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr2 + (0))
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp22 = tl.load(in_ptr4 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp27 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp135 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp171 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp189 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp207 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp225 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp243 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp261 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp279 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp0 / tmp2
    tmp6 = tmp3 - tmp5
    tmp7 = tl.full([1], 0, tl.int32)
    tmp8 = tmp7 < tmp6
    tmp9 = tmp8.to(tl.int8)
    tmp10 = tmp6 < tmp7
    tmp11 = tmp10.to(tl.int8)
    tmp12 = tmp9 - tmp11
    tmp13 = tmp12.to(tmp6.dtype)
    tmp14 = tl_math.abs(tmp6)
    tmp15 = 0.5
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.floor(tmp16)
    tmp18 = tmp13 * tmp17
    tmp21 = triton_helpers.maximum(tmp18, tmp20)
    tmp24 = triton_helpers.minimum(tmp21, tmp23)
    tmp25 = tmp24 + tmp5
    tmp26 = tmp25 * tmp2
    tmp28 = tmp27 / tmp2
    tmp29 = tmp28 - tmp5
    tmp30 = tmp7 < tmp29
    tmp31 = tmp30.to(tl.int8)
    tmp32 = tmp29 < tmp7
    tmp33 = tmp32.to(tl.int8)
    tmp34 = tmp31 - tmp33
    tmp35 = tmp34.to(tmp29.dtype)
    tmp36 = tl_math.abs(tmp29)
    tmp37 = tmp36 + tmp15
    tmp38 = libdevice.floor(tmp37)
    tmp39 = tmp35 * tmp38
    tmp40 = triton_helpers.maximum(tmp39, tmp20)
    tmp41 = triton_helpers.minimum(tmp40, tmp23)
    tmp42 = tmp41 + tmp5
    tmp43 = tmp42 * tmp2
    tmp44 = tmp43 + tmp26
    tmp46 = tmp45 / tmp2
    tmp47 = tmp46 - tmp5
    tmp48 = tmp7 < tmp47
    tmp49 = tmp48.to(tl.int8)
    tmp50 = tmp47 < tmp7
    tmp51 = tmp50.to(tl.int8)
    tmp52 = tmp49 - tmp51
    tmp53 = tmp52.to(tmp47.dtype)
    tmp54 = tl_math.abs(tmp47)
    tmp55 = tmp54 + tmp15
    tmp56 = libdevice.floor(tmp55)
    tmp57 = tmp53 * tmp56
    tmp58 = triton_helpers.maximum(tmp57, tmp20)
    tmp59 = triton_helpers.minimum(tmp58, tmp23)
    tmp60 = tmp59 + tmp5
    tmp61 = tmp60 * tmp2
    tmp62 = tmp61 + tmp44
    tmp64 = tmp63 / tmp2
    tmp65 = tmp64 - tmp5
    tmp66 = tmp7 < tmp65
    tmp67 = tmp66.to(tl.int8)
    tmp68 = tmp65 < tmp7
    tmp69 = tmp68.to(tl.int8)
    tmp70 = tmp67 - tmp69
    tmp71 = tmp70.to(tmp65.dtype)
    tmp72 = tl_math.abs(tmp65)
    tmp73 = tmp72 + tmp15
    tmp74 = libdevice.floor(tmp73)
    tmp75 = tmp71 * tmp74
    tmp76 = triton_helpers.maximum(tmp75, tmp20)
    tmp77 = triton_helpers.minimum(tmp76, tmp23)
    tmp78 = tmp77 + tmp5
    tmp79 = tmp78 * tmp2
    tmp80 = tmp79 + tmp62
    tmp82 = tmp81 / tmp2
    tmp83 = tmp82 - tmp5
    tmp84 = tmp7 < tmp83
    tmp85 = tmp84.to(tl.int8)
    tmp86 = tmp83 < tmp7
    tmp87 = tmp86.to(tl.int8)
    tmp88 = tmp85 - tmp87
    tmp89 = tmp88.to(tmp83.dtype)
    tmp90 = tl_math.abs(tmp83)
    tmp91 = tmp90 + tmp15
    tmp92 = libdevice.floor(tmp91)
    tmp93 = tmp89 * tmp92
    tmp94 = triton_helpers.maximum(tmp93, tmp20)
    tmp95 = triton_helpers.minimum(tmp94, tmp23)
    tmp96 = tmp95 + tmp5
    tmp97 = tmp96 * tmp2
    tmp98 = tmp97 + tmp80
    tmp100 = tmp99 / tmp2
    tmp101 = tmp100 - tmp5
    tmp102 = tmp7 < tmp101
    tmp103 = tmp102.to(tl.int8)
    tmp104 = tmp101 < tmp7
    tmp105 = tmp104.to(tl.int8)
    tmp106 = tmp103 - tmp105
    tmp107 = tmp106.to(tmp101.dtype)
    tmp108 = tl_math.abs(tmp101)
    tmp109 = tmp108 + tmp15
    tmp110 = libdevice.floor(tmp109)
    tmp111 = tmp107 * tmp110
    tmp112 = triton_helpers.maximum(tmp111, tmp20)
    tmp113 = triton_helpers.minimum(tmp112, tmp23)
    tmp114 = tmp113 + tmp5
    tmp115 = tmp114 * tmp2
    tmp116 = tmp115 + tmp98
    tmp118 = tmp117 / tmp2
    tmp119 = tmp118 - tmp5
    tmp120 = tmp7 < tmp119
    tmp121 = tmp120.to(tl.int8)
    tmp122 = tmp119 < tmp7
    tmp123 = tmp122.to(tl.int8)
    tmp124 = tmp121 - tmp123
    tmp125 = tmp124.to(tmp119.dtype)
    tmp126 = tl_math.abs(tmp119)
    tmp127 = tmp126 + tmp15
    tmp128 = libdevice.floor(tmp127)
    tmp129 = tmp125 * tmp128
    tmp130 = triton_helpers.maximum(tmp129, tmp20)
    tmp131 = triton_helpers.minimum(tmp130, tmp23)
    tmp132 = tmp131 + tmp5
    tmp133 = tmp132 * tmp2
    tmp134 = tmp133 + tmp116
    tmp136 = tmp135 / tmp2
    tmp137 = tmp136 - tmp5
    tmp138 = tmp7 < tmp137
    tmp139 = tmp138.to(tl.int8)
    tmp140 = tmp137 < tmp7
    tmp141 = tmp140.to(tl.int8)
    tmp142 = tmp139 - tmp141
    tmp143 = tmp142.to(tmp137.dtype)
    tmp144 = tl_math.abs(tmp137)
    tmp145 = tmp144 + tmp15
    tmp146 = libdevice.floor(tmp145)
    tmp147 = tmp143 * tmp146
    tmp148 = triton_helpers.maximum(tmp147, tmp20)
    tmp149 = triton_helpers.minimum(tmp148, tmp23)
    tmp150 = tmp149 + tmp5
    tmp151 = tmp150 * tmp2
    tmp152 = tmp151 + tmp134
    tmp154 = tmp153 / tmp2
    tmp155 = tmp154 - tmp5
    tmp156 = tmp7 < tmp155
    tmp157 = tmp156.to(tl.int8)
    tmp158 = tmp155 < tmp7
    tmp159 = tmp158.to(tl.int8)
    tmp160 = tmp157 - tmp159
    tmp161 = tmp160.to(tmp155.dtype)
    tmp162 = tl_math.abs(tmp155)
    tmp163 = tmp162 + tmp15
    tmp164 = libdevice.floor(tmp163)
    tmp165 = tmp161 * tmp164
    tmp166 = triton_helpers.maximum(tmp165, tmp20)
    tmp167 = triton_helpers.minimum(tmp166, tmp23)
    tmp168 = tmp167 + tmp5
    tmp169 = tmp168 * tmp2
    tmp170 = tmp169 + tmp152
    tmp172 = tmp171 / tmp2
    tmp173 = tmp172 - tmp5
    tmp174 = tmp7 < tmp173
    tmp175 = tmp174.to(tl.int8)
    tmp176 = tmp173 < tmp7
    tmp177 = tmp176.to(tl.int8)
    tmp178 = tmp175 - tmp177
    tmp179 = tmp178.to(tmp173.dtype)
    tmp180 = tl_math.abs(tmp173)
    tmp181 = tmp180 + tmp15
    tmp182 = libdevice.floor(tmp181)
    tmp183 = tmp179 * tmp182
    tmp184 = triton_helpers.maximum(tmp183, tmp20)
    tmp185 = triton_helpers.minimum(tmp184, tmp23)
    tmp186 = tmp185 + tmp5
    tmp187 = tmp186 * tmp2
    tmp188 = tmp187 + tmp170
    tmp190 = tmp189 / tmp2
    tmp191 = tmp190 - tmp5
    tmp192 = tmp7 < tmp191
    tmp193 = tmp192.to(tl.int8)
    tmp194 = tmp191 < tmp7
    tmp195 = tmp194.to(tl.int8)
    tmp196 = tmp193 - tmp195
    tmp197 = tmp196.to(tmp191.dtype)
    tmp198 = tl_math.abs(tmp191)
    tmp199 = tmp198 + tmp15
    tmp200 = libdevice.floor(tmp199)
    tmp201 = tmp197 * tmp200
    tmp202 = triton_helpers.maximum(tmp201, tmp20)
    tmp203 = triton_helpers.minimum(tmp202, tmp23)
    tmp204 = tmp203 + tmp5
    tmp205 = tmp204 * tmp2
    tmp206 = tmp205 + tmp188
    tmp208 = tmp207 / tmp2
    tmp209 = tmp208 - tmp5
    tmp210 = tmp7 < tmp209
    tmp211 = tmp210.to(tl.int8)
    tmp212 = tmp209 < tmp7
    tmp213 = tmp212.to(tl.int8)
    tmp214 = tmp211 - tmp213
    tmp215 = tmp214.to(tmp209.dtype)
    tmp216 = tl_math.abs(tmp209)
    tmp217 = tmp216 + tmp15
    tmp218 = libdevice.floor(tmp217)
    tmp219 = tmp215 * tmp218
    tmp220 = triton_helpers.maximum(tmp219, tmp20)
    tmp221 = triton_helpers.minimum(tmp220, tmp23)
    tmp222 = tmp221 + tmp5
    tmp223 = tmp222 * tmp2
    tmp224 = tmp223 + tmp206
    tmp226 = tmp225 / tmp2
    tmp227 = tmp226 - tmp5
    tmp228 = tmp7 < tmp227
    tmp229 = tmp228.to(tl.int8)
    tmp230 = tmp227 < tmp7
    tmp231 = tmp230.to(tl.int8)
    tmp232 = tmp229 - tmp231
    tmp233 = tmp232.to(tmp227.dtype)
    tmp234 = tl_math.abs(tmp227)
    tmp235 = tmp234 + tmp15
    tmp236 = libdevice.floor(tmp235)
    tmp237 = tmp233 * tmp236
    tmp238 = triton_helpers.maximum(tmp237, tmp20)
    tmp239 = triton_helpers.minimum(tmp238, tmp23)
    tmp240 = tmp239 + tmp5
    tmp241 = tmp240 * tmp2
    tmp242 = tmp241 + tmp224
    tmp244 = tmp243 / tmp2
    tmp245 = tmp244 - tmp5
    tmp246 = tmp7 < tmp245
    tmp247 = tmp246.to(tl.int8)
    tmp248 = tmp245 < tmp7
    tmp249 = tmp248.to(tl.int8)
    tmp250 = tmp247 - tmp249
    tmp251 = tmp250.to(tmp245.dtype)
    tmp252 = tl_math.abs(tmp245)
    tmp253 = tmp252 + tmp15
    tmp254 = libdevice.floor(tmp253)
    tmp255 = tmp251 * tmp254
    tmp256 = triton_helpers.maximum(tmp255, tmp20)
    tmp257 = triton_helpers.minimum(tmp256, tmp23)
    tmp258 = tmp257 + tmp5
    tmp259 = tmp258 * tmp2
    tmp260 = tmp259 + tmp242
    tmp262 = tmp261 / tmp2
    tmp263 = tmp262 - tmp5
    tmp264 = tmp7 < tmp263
    tmp265 = tmp264.to(tl.int8)
    tmp266 = tmp263 < tmp7
    tmp267 = tmp266.to(tl.int8)
    tmp268 = tmp265 - tmp267
    tmp269 = tmp268.to(tmp263.dtype)
    tmp270 = tl_math.abs(tmp263)
    tmp271 = tmp270 + tmp15
    tmp272 = libdevice.floor(tmp271)
    tmp273 = tmp269 * tmp272
    tmp274 = triton_helpers.maximum(tmp273, tmp20)
    tmp275 = triton_helpers.minimum(tmp274, tmp23)
    tmp276 = tmp275 + tmp5
    tmp277 = tmp276 * tmp2
    tmp278 = tmp277 + tmp260
    tmp280 = tmp279 / tmp2
    tmp281 = tmp280 - tmp5
    tmp282 = tmp7 < tmp281
    tmp283 = tmp282.to(tl.int8)
    tmp284 = tmp281 < tmp7
    tmp285 = tmp284.to(tl.int8)
    tmp286 = tmp283 - tmp285
    tmp287 = tmp286.to(tmp281.dtype)
    tmp288 = tl_math.abs(tmp281)
    tmp289 = tmp288 + tmp15
    tmp290 = libdevice.floor(tmp289)
    tmp291 = tmp287 * tmp290
    tmp292 = triton_helpers.maximum(tmp291, tmp20)
    tmp293 = triton_helpers.minimum(tmp292, tmp23)
    tmp294 = tmp293 + tmp5
    tmp295 = tmp294 * tmp2
    tmp296 = tmp295 + tmp278
    tmp297 = 0.0625
    tmp298 = tmp296 * tmp297
    tl.store(out_ptr0 + (x0), tmp298, xmask)
