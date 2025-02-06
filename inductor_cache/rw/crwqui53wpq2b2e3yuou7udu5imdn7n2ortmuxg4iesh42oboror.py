
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*i64', 'in_ptr15': '*i64', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 76, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_linalg_vector_norm_sub_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp7 = tl.load(in_ptr2 + (x0 + 64*x1), xmask)
    tmp13 = tl.load(in_ptr4 + (x0 + 64*x1), xmask)
    tmp15 = tl.load(in_ptr5 + (x0 + 64*x1), xmask)
    tmp17 = tl.load(in_ptr6 + (x0 + 64*x1), xmask)
    tmp19 = tl.load(in_ptr7 + (x0 + 64*x1), xmask)
    tmp23 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp24 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp29 = tl.load(in_ptr2 + (16 + x0 + 64*x1), xmask)
    tmp35 = tl.load(in_ptr4 + (16 + x0 + 64*x1), xmask)
    tmp37 = tl.load(in_ptr5 + (16 + x0 + 64*x1), xmask)
    tmp39 = tl.load(in_ptr6 + (16 + x0 + 64*x1), xmask)
    tmp41 = tl.load(in_ptr7 + (16 + x0 + 64*x1), xmask)
    tmp46 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp47 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp52 = tl.load(in_ptr2 + (32 + x0 + 64*x1), xmask)
    tmp58 = tl.load(in_ptr4 + (32 + x0 + 64*x1), xmask)
    tmp60 = tl.load(in_ptr5 + (32 + x0 + 64*x1), xmask)
    tmp62 = tl.load(in_ptr6 + (32 + x0 + 64*x1), xmask)
    tmp64 = tl.load(in_ptr7 + (32 + x0 + 64*x1), xmask)
    tmp69 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
    tmp70 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp75 = tl.load(in_ptr2 + (48 + x0 + 64*x1), xmask)
    tmp81 = tl.load(in_ptr4 + (48 + x0 + 64*x1), xmask)
    tmp83 = tl.load(in_ptr5 + (48 + x0 + 64*x1), xmask)
    tmp85 = tl.load(in_ptr6 + (48 + x0 + 64*x1), xmask)
    tmp87 = tl.load(in_ptr7 + (48 + x0 + 64*x1), xmask)
    tmp92 = tl.load(in_ptr8 + (x0 + 64*x1), xmask)
    tmp97 = tl.load(in_ptr9 + (x0 + 64*x1), xmask)
    tmp103 = tl.load(in_ptr10 + (x0 + 64*x1), xmask)
    tmp105 = tl.load(in_ptr11 + (x0 + 64*x1), xmask)
    tmp107 = tl.load(in_ptr12 + (x0 + 64*x1), xmask)
    tmp109 = tl.load(in_ptr13 + (x0 + 64*x1), xmask)
    tmp113 = tl.load(in_ptr8 + (16 + x0 + 64*x1), xmask)
    tmp118 = tl.load(in_ptr9 + (16 + x0 + 64*x1), xmask)
    tmp124 = tl.load(in_ptr10 + (16 + x0 + 64*x1), xmask)
    tmp126 = tl.load(in_ptr11 + (16 + x0 + 64*x1), xmask)
    tmp128 = tl.load(in_ptr12 + (16 + x0 + 64*x1), xmask)
    tmp130 = tl.load(in_ptr13 + (16 + x0 + 64*x1), xmask)
    tmp135 = tl.load(in_ptr8 + (32 + x0 + 64*x1), xmask)
    tmp140 = tl.load(in_ptr9 + (32 + x0 + 64*x1), xmask)
    tmp146 = tl.load(in_ptr10 + (32 + x0 + 64*x1), xmask)
    tmp148 = tl.load(in_ptr11 + (32 + x0 + 64*x1), xmask)
    tmp150 = tl.load(in_ptr12 + (32 + x0 + 64*x1), xmask)
    tmp152 = tl.load(in_ptr13 + (32 + x0 + 64*x1), xmask)
    tmp157 = tl.load(in_ptr8 + (48 + x0 + 64*x1), xmask)
    tmp162 = tl.load(in_ptr9 + (48 + x0 + 64*x1), xmask)
    tmp168 = tl.load(in_ptr10 + (48 + x0 + 64*x1), xmask)
    tmp170 = tl.load(in_ptr11 + (48 + x0 + 64*x1), xmask)
    tmp172 = tl.load(in_ptr12 + (48 + x0 + 64*x1), xmask)
    tmp174 = tl.load(in_ptr13 + (48 + x0 + 64*x1), xmask)
    tmp179 = tl.load(in_ptr14 + (x0 + 64*x1), xmask)
    tmp184 = tl.load(in_ptr15 + (x0 + 64*x1), xmask)
    tmp190 = tl.load(in_ptr16 + (x0 + 64*x1), xmask)
    tmp192 = tl.load(in_ptr17 + (x0 + 64*x1), xmask)
    tmp194 = tl.load(in_ptr18 + (x0 + 64*x1), xmask)
    tmp196 = tl.load(in_ptr19 + (x0 + 64*x1), xmask)
    tmp200 = tl.load(in_ptr14 + (16 + x0 + 64*x1), xmask)
    tmp205 = tl.load(in_ptr15 + (16 + x0 + 64*x1), xmask)
    tmp211 = tl.load(in_ptr16 + (16 + x0 + 64*x1), xmask)
    tmp213 = tl.load(in_ptr17 + (16 + x0 + 64*x1), xmask)
    tmp215 = tl.load(in_ptr18 + (16 + x0 + 64*x1), xmask)
    tmp217 = tl.load(in_ptr19 + (16 + x0 + 64*x1), xmask)
    tmp222 = tl.load(in_ptr14 + (32 + x0 + 64*x1), xmask)
    tmp227 = tl.load(in_ptr15 + (32 + x0 + 64*x1), xmask)
    tmp233 = tl.load(in_ptr16 + (32 + x0 + 64*x1), xmask)
    tmp235 = tl.load(in_ptr17 + (32 + x0 + 64*x1), xmask)
    tmp237 = tl.load(in_ptr18 + (32 + x0 + 64*x1), xmask)
    tmp239 = tl.load(in_ptr19 + (32 + x0 + 64*x1), xmask)
    tmp244 = tl.load(in_ptr14 + (48 + x0 + 64*x1), xmask)
    tmp249 = tl.load(in_ptr15 + (48 + x0 + 64*x1), xmask)
    tmp255 = tl.load(in_ptr16 + (48 + x0 + 64*x1), xmask)
    tmp257 = tl.load(in_ptr17 + (48 + x0 + 64*x1), xmask)
    tmp259 = tl.load(in_ptr18 + (48 + x0 + 64*x1), xmask)
    tmp261 = tl.load(in_ptr19 + (48 + x0 + 64*x1), xmask)
    tmp2 = tl.full([XBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp5 < 4")
    tmp8 = tmp7 + tmp2
    tmp9 = tmp7 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp7)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp10 < 4")
    tmp12 = tl.load(in_ptr3 + (tmp10 + 4*tmp5 + 64*x1), xmask, eviction_policy='evict_last')
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp18 = tmp16 + tmp17
    tmp20 = tmp18 + tmp19
    tmp21 = tmp0 - tmp20
    tmp22 = tl_math.abs(tmp21)
    tmp25 = tmp24 + tmp2
    tmp26 = tmp24 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp24)
    tl.device_assert(((0 <= tmp27) & (tmp27 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp27 < 4")
    tmp30 = tmp29 + tmp2
    tmp31 = tmp29 < 0
    tmp32 = tl.where(tmp31, tmp30, tmp29)
    tl.device_assert(((0 <= tmp32) & (tmp32 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp32 < 4")
    tmp34 = tl.load(in_ptr3 + (16 + tmp32 + 4*tmp27 + 64*x1), xmask, eviction_policy='evict_last')
    tmp36 = tmp34 * tmp35
    tmp38 = tmp36 + tmp37
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp23 - tmp42
    tmp44 = tl_math.abs(tmp43)
    tmp45 = tmp22 + tmp44
    tmp48 = tmp47 + tmp2
    tmp49 = tmp47 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp47)
    tl.device_assert(((0 <= tmp50) & (tmp50 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp50 < 4")
    tmp53 = tmp52 + tmp2
    tmp54 = tmp52 < 0
    tmp55 = tl.where(tmp54, tmp53, tmp52)
    tl.device_assert(((0 <= tmp55) & (tmp55 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp55 < 4")
    tmp57 = tl.load(in_ptr3 + (32 + tmp55 + 4*tmp50 + 64*x1), xmask, eviction_policy='evict_last')
    tmp59 = tmp57 * tmp58
    tmp61 = tmp59 + tmp60
    tmp63 = tmp61 + tmp62
    tmp65 = tmp63 + tmp64
    tmp66 = tmp46 - tmp65
    tmp67 = tl_math.abs(tmp66)
    tmp68 = tmp45 + tmp67
    tmp71 = tmp70 + tmp2
    tmp72 = tmp70 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp70)
    tl.device_assert(((0 <= tmp73) & (tmp73 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp73 < 4")
    tmp76 = tmp75 + tmp2
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tl.device_assert(((0 <= tmp78) & (tmp78 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp78 < 4")
    tmp80 = tl.load(in_ptr3 + (48 + tmp78 + 4*tmp73 + 64*x1), xmask, eviction_policy='evict_last')
    tmp82 = tmp80 * tmp81
    tmp84 = tmp82 + tmp83
    tmp86 = tmp84 + tmp85
    tmp88 = tmp86 + tmp87
    tmp89 = tmp69 - tmp88
    tmp90 = tl_math.abs(tmp89)
    tmp91 = tmp68 + tmp90
    tmp93 = tmp92 + tmp2
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tl.device_assert(((0 <= tmp95) & (tmp95 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp95 < 4")
    tmp98 = tmp97 + tmp2
    tmp99 = tmp97 < 0
    tmp100 = tl.where(tmp99, tmp98, tmp97)
    tl.device_assert(((0 <= tmp100) & (tmp100 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp100 < 4")
    tmp102 = tl.load(in_ptr3 + (tmp100 + 4*tmp95 + 64*x1), xmask, eviction_policy='evict_last')
    tmp104 = tmp102 * tmp103
    tmp106 = tmp104 + tmp105
    tmp108 = tmp106 + tmp107
    tmp110 = tmp108 + tmp109
    tmp111 = tmp0 - tmp110
    tmp112 = tl_math.abs(tmp111)
    tmp114 = tmp113 + tmp2
    tmp115 = tmp113 < 0
    tmp116 = tl.where(tmp115, tmp114, tmp113)
    tl.device_assert(((0 <= tmp116) & (tmp116 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp116 < 4")
    tmp119 = tmp118 + tmp2
    tmp120 = tmp118 < 0
    tmp121 = tl.where(tmp120, tmp119, tmp118)
    tl.device_assert(((0 <= tmp121) & (tmp121 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp121 < 4")
    tmp123 = tl.load(in_ptr3 + (16 + tmp121 + 4*tmp116 + 64*x1), xmask, eviction_policy='evict_last')
    tmp125 = tmp123 * tmp124
    tmp127 = tmp125 + tmp126
    tmp129 = tmp127 + tmp128
    tmp131 = tmp129 + tmp130
    tmp132 = tmp23 - tmp131
    tmp133 = tl_math.abs(tmp132)
    tmp134 = tmp112 + tmp133
    tmp136 = tmp135 + tmp2
    tmp137 = tmp135 < 0
    tmp138 = tl.where(tmp137, tmp136, tmp135)
    tl.device_assert(((0 <= tmp138) & (tmp138 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp138 < 4")
    tmp141 = tmp140 + tmp2
    tmp142 = tmp140 < 0
    tmp143 = tl.where(tmp142, tmp141, tmp140)
    tl.device_assert(((0 <= tmp143) & (tmp143 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp143 < 4")
    tmp145 = tl.load(in_ptr3 + (32 + tmp143 + 4*tmp138 + 64*x1), xmask, eviction_policy='evict_last')
    tmp147 = tmp145 * tmp146
    tmp149 = tmp147 + tmp148
    tmp151 = tmp149 + tmp150
    tmp153 = tmp151 + tmp152
    tmp154 = tmp46 - tmp153
    tmp155 = tl_math.abs(tmp154)
    tmp156 = tmp134 + tmp155
    tmp158 = tmp157 + tmp2
    tmp159 = tmp157 < 0
    tmp160 = tl.where(tmp159, tmp158, tmp157)
    tl.device_assert(((0 <= tmp160) & (tmp160 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp160 < 4")
    tmp163 = tmp162 + tmp2
    tmp164 = tmp162 < 0
    tmp165 = tl.where(tmp164, tmp163, tmp162)
    tl.device_assert(((0 <= tmp165) & (tmp165 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp165 < 4")
    tmp167 = tl.load(in_ptr3 + (48 + tmp165 + 4*tmp160 + 64*x1), xmask, eviction_policy='evict_last')
    tmp169 = tmp167 * tmp168
    tmp171 = tmp169 + tmp170
    tmp173 = tmp171 + tmp172
    tmp175 = tmp173 + tmp174
    tmp176 = tmp69 - tmp175
    tmp177 = tl_math.abs(tmp176)
    tmp178 = tmp156 + tmp177
    tmp180 = tmp179 + tmp2
    tmp181 = tmp179 < 0
    tmp182 = tl.where(tmp181, tmp180, tmp179)
    tl.device_assert(((0 <= tmp182) & (tmp182 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp182 < 4")
    tmp185 = tmp184 + tmp2
    tmp186 = tmp184 < 0
    tmp187 = tl.where(tmp186, tmp185, tmp184)
    tl.device_assert(((0 <= tmp187) & (tmp187 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp187 < 4")
    tmp189 = tl.load(in_ptr3 + (tmp187 + 4*tmp182 + 64*x1), xmask, eviction_policy='evict_last')
    tmp191 = tmp189 * tmp190
    tmp193 = tmp191 + tmp192
    tmp195 = tmp193 + tmp194
    tmp197 = tmp195 + tmp196
    tmp198 = tmp0 - tmp197
    tmp199 = tl_math.abs(tmp198)
    tmp201 = tmp200 + tmp2
    tmp202 = tmp200 < 0
    tmp203 = tl.where(tmp202, tmp201, tmp200)
    tl.device_assert(((0 <= tmp203) & (tmp203 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp203 < 4")
    tmp206 = tmp205 + tmp2
    tmp207 = tmp205 < 0
    tmp208 = tl.where(tmp207, tmp206, tmp205)
    tl.device_assert(((0 <= tmp208) & (tmp208 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp208 < 4")
    tmp210 = tl.load(in_ptr3 + (16 + tmp208 + 4*tmp203 + 64*x1), xmask, eviction_policy='evict_last')
    tmp212 = tmp210 * tmp211
    tmp214 = tmp212 + tmp213
    tmp216 = tmp214 + tmp215
    tmp218 = tmp216 + tmp217
    tmp219 = tmp23 - tmp218
    tmp220 = tl_math.abs(tmp219)
    tmp221 = tmp199 + tmp220
    tmp223 = tmp222 + tmp2
    tmp224 = tmp222 < 0
    tmp225 = tl.where(tmp224, tmp223, tmp222)
    tl.device_assert(((0 <= tmp225) & (tmp225 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp225 < 4")
    tmp228 = tmp227 + tmp2
    tmp229 = tmp227 < 0
    tmp230 = tl.where(tmp229, tmp228, tmp227)
    tl.device_assert(((0 <= tmp230) & (tmp230 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp230 < 4")
    tmp232 = tl.load(in_ptr3 + (32 + tmp230 + 4*tmp225 + 64*x1), xmask, eviction_policy='evict_last')
    tmp234 = tmp232 * tmp233
    tmp236 = tmp234 + tmp235
    tmp238 = tmp236 + tmp237
    tmp240 = tmp238 + tmp239
    tmp241 = tmp46 - tmp240
    tmp242 = tl_math.abs(tmp241)
    tmp243 = tmp221 + tmp242
    tmp245 = tmp244 + tmp2
    tmp246 = tmp244 < 0
    tmp247 = tl.where(tmp246, tmp245, tmp244)
    tl.device_assert(((0 <= tmp247) & (tmp247 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp247 < 4")
    tmp250 = tmp249 + tmp2
    tmp251 = tmp249 < 0
    tmp252 = tl.where(tmp251, tmp250, tmp249)
    tl.device_assert(((0 <= tmp252) & (tmp252 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp252 < 4")
    tmp254 = tl.load(in_ptr3 + (48 + tmp252 + 4*tmp247 + 64*x1), xmask, eviction_policy='evict_last')
    tmp256 = tmp254 * tmp255
    tmp258 = tmp256 + tmp257
    tmp260 = tmp258 + tmp259
    tmp262 = tmp260 + tmp261
    tmp263 = tmp69 - tmp262
    tmp264 = tl_math.abs(tmp263)
    tmp265 = tmp243 + tmp264
    tl.store(out_ptr0 + (x2), tmp91, xmask)
    tl.store(out_ptr1 + (x2), tmp178, xmask)
    tl.store(out_ptr2 + (x2), tmp265, xmask)
