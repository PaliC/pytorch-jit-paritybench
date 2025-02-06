
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'in_ptr30': '*fp32', 'in_ptr31': '*fp32', 'in_ptr32': '*fp32', 'in_ptr33': '*fp32', 'in_ptr34': '*fp32', 'in_ptr35': '*fp32', 'in_ptr36': '*fp32', 'in_ptr37': '*fp32', 'in_ptr38': '*fp32', 'in_ptr39': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 40, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 256)
    x4 = xindex // 256
    x3 = xindex // 16384
    x5 = ((xindex // 256) % 64)
    x1 = ((xindex // 256) % 8)
    x6 = xindex // 2048
    x2 = ((xindex // 2048) % 8)
    x7 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = 0.0
    tmp22 = triton_helpers.maximum(tmp20, tmp21)
    tmp23 = 6.0
    tmp24 = triton_helpers.minimum(tmp22, tmp23)
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp4, tmp24, tmp25)
    tmp27 = tmp0 >= tmp3
    tmp28 = tl.full([1], 64, tl.int64)
    tmp29 = tmp0 < tmp28
    tmp30 = tmp27 & tmp29
    tmp31 = tl.load(in_ptr5 + (32*x4 + ((-32) + x0)), tmp30, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr6 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 - tmp32
    tmp34 = tl.load(in_ptr7 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.sqrt(tmp36)
    tmp38 = tl.full([1], 1, tl.int32)
    tmp39 = tmp38 / tmp37
    tmp40 = 1.0
    tmp41 = tmp39 * tmp40
    tmp42 = tmp33 * tmp41
    tmp43 = tl.load(in_ptr8 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 * tmp43
    tmp45 = tl.load(in_ptr9 + ((-32) + x0), tmp30, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp44 + tmp45
    tmp47 = 0.0
    tmp48 = triton_helpers.maximum(tmp46, tmp47)
    tmp49 = 6.0
    tmp50 = triton_helpers.minimum(tmp48, tmp49)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp30, tmp50, tmp51)
    tmp53 = tmp0 >= tmp28
    tmp54 = tl.full([1], 96, tl.int64)
    tmp55 = tmp0 < tmp54
    tmp56 = tmp53 & tmp55
    tmp57 = tl.load(in_ptr10 + (32*x4 + ((-64) + x0)), tmp56, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.load(in_ptr11 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 - tmp58
    tmp60 = tl.load(in_ptr12 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp61 = 1e-05
    tmp62 = tmp60 + tmp61
    tmp63 = libdevice.sqrt(tmp62)
    tmp64 = tl.full([1], 1, tl.int32)
    tmp65 = tmp64 / tmp63
    tmp66 = 1.0
    tmp67 = tmp65 * tmp66
    tmp68 = tmp59 * tmp67
    tmp69 = tl.load(in_ptr13 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp68 * tmp69
    tmp71 = tl.load(in_ptr14 + ((-64) + x0), tmp56, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 + tmp71
    tmp73 = 0.0
    tmp74 = triton_helpers.maximum(tmp72, tmp73)
    tmp75 = 6.0
    tmp76 = triton_helpers.minimum(tmp74, tmp75)
    tmp77 = tl.full(tmp76.shape, 0.0, tmp76.dtype)
    tmp78 = tl.where(tmp56, tmp76, tmp77)
    tmp79 = tmp0 >= tmp54
    tmp80 = tl.full([1], 128, tl.int64)
    tmp81 = tmp0 < tmp80
    tmp82 = tmp79 & tmp81
    tmp83 = tl.load(in_ptr15 + (32*x4 + ((-96) + x0)), tmp82, eviction_policy='evict_last', other=0.0)
    tmp84 = tl.load(in_ptr16 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 - tmp84
    tmp86 = tl.load(in_ptr17 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = libdevice.sqrt(tmp88)
    tmp90 = tl.full([1], 1, tl.int32)
    tmp91 = tmp90 / tmp89
    tmp92 = 1.0
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 * tmp93
    tmp95 = tl.load(in_ptr18 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp96 = tmp94 * tmp95
    tmp97 = tl.load(in_ptr19 + ((-96) + x0), tmp82, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp96 + tmp97
    tmp99 = 0.0
    tmp100 = triton_helpers.maximum(tmp98, tmp99)
    tmp101 = 6.0
    tmp102 = triton_helpers.minimum(tmp100, tmp101)
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp82, tmp102, tmp103)
    tmp105 = tmp0 >= tmp80
    tmp106 = tl.full([1], 160, tl.int64)
    tmp107 = tmp0 < tmp106
    tmp108 = tmp105 & tmp107
    tmp109 = tl.load(in_ptr20 + (32*x5 + 2304*x3 + ((-128) + x0)), tmp108, eviction_policy='evict_last', other=0.0)
    tmp110 = tl.load(in_ptr21 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp109 - tmp110
    tmp112 = tl.load(in_ptr22 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp113 = 1e-05
    tmp114 = tmp112 + tmp113
    tmp115 = libdevice.sqrt(tmp114)
    tmp116 = tl.full([1], 1, tl.int32)
    tmp117 = tmp116 / tmp115
    tmp118 = 1.0
    tmp119 = tmp117 * tmp118
    tmp120 = tmp111 * tmp119
    tmp121 = tl.load(in_ptr23 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp122 = tmp120 * tmp121
    tmp123 = tl.load(in_ptr24 + ((-128) + x0), tmp108, eviction_policy='evict_last', other=0.0)
    tmp124 = tmp122 + tmp123
    tmp125 = 0.0
    tmp126 = triton_helpers.maximum(tmp124, tmp125)
    tmp127 = 6.0
    tmp128 = triton_helpers.minimum(tmp126, tmp127)
    tmp129 = tl.full(tmp128.shape, 0.0, tmp128.dtype)
    tmp130 = tl.where(tmp108, tmp128, tmp129)
    tmp131 = tmp0 >= tmp106
    tmp132 = tl.full([1], 192, tl.int64)
    tmp133 = tmp0 < tmp132
    tmp134 = tmp131 & tmp133
    tmp135 = tl.load(in_ptr25 + (32*x1 + 288*x6 + ((-160) + x0)), tmp134, eviction_policy='evict_last', other=0.0)
    tmp136 = tl.load(in_ptr26 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp137 = tmp135 - tmp136
    tmp138 = tl.load(in_ptr27 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp139 = 1e-05
    tmp140 = tmp138 + tmp139
    tmp141 = libdevice.sqrt(tmp140)
    tmp142 = tl.full([1], 1, tl.int32)
    tmp143 = tmp142 / tmp141
    tmp144 = 1.0
    tmp145 = tmp143 * tmp144
    tmp146 = tmp137 * tmp145
    tmp147 = tl.load(in_ptr28 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp148 = tmp146 * tmp147
    tmp149 = tl.load(in_ptr29 + ((-160) + x0), tmp134, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp148 + tmp149
    tmp151 = 0.0
    tmp152 = triton_helpers.maximum(tmp150, tmp151)
    tmp153 = 6.0
    tmp154 = triton_helpers.minimum(tmp152, tmp153)
    tmp155 = tl.full(tmp154.shape, 0.0, tmp154.dtype)
    tmp156 = tl.where(tmp134, tmp154, tmp155)
    tmp157 = tmp0 >= tmp132
    tmp158 = tl.full([1], 224, tl.int64)
    tmp159 = tmp0 < tmp158
    tmp160 = tmp157 & tmp159
    tmp161 = tl.load(in_ptr30 + (32*x1 + 288*x2 + 2592*x3 + ((-192) + x0)), tmp160, eviction_policy='evict_last', other=0.0)
    tmp162 = tl.load(in_ptr31 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp163 = tmp161 - tmp162
    tmp164 = tl.load(in_ptr32 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp165 = 1e-05
    tmp166 = tmp164 + tmp165
    tmp167 = libdevice.sqrt(tmp166)
    tmp168 = tl.full([1], 1, tl.int32)
    tmp169 = tmp168 / tmp167
    tmp170 = 1.0
    tmp171 = tmp169 * tmp170
    tmp172 = tmp163 * tmp171
    tmp173 = tl.load(in_ptr33 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp174 = tmp172 * tmp173
    tmp175 = tl.load(in_ptr34 + ((-192) + x0), tmp160, eviction_policy='evict_last', other=0.0)
    tmp176 = tmp174 + tmp175
    tmp177 = 0.0
    tmp178 = triton_helpers.maximum(tmp176, tmp177)
    tmp179 = 6.0
    tmp180 = triton_helpers.minimum(tmp178, tmp179)
    tmp181 = tl.full(tmp180.shape, 0.0, tmp180.dtype)
    tmp182 = tl.where(tmp160, tmp180, tmp181)
    tmp183 = tmp0 >= tmp158
    tmp184 = tl.full([1], 256, tl.int64)
    tmp185 = tmp0 < tmp184
    tmp186 = tl.load(in_ptr35 + (32*x4 + ((-224) + x0)), tmp183, eviction_policy='evict_last', other=0.0)
    tmp187 = tl.load(in_ptr36 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp188 = tmp186 - tmp187
    tmp189 = tl.load(in_ptr37 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp190 = 1e-05
    tmp191 = tmp189 + tmp190
    tmp192 = libdevice.sqrt(tmp191)
    tmp193 = tl.full([1], 1, tl.int32)
    tmp194 = tmp193 / tmp192
    tmp195 = 1.0
    tmp196 = tmp194 * tmp195
    tmp197 = tmp188 * tmp196
    tmp198 = tl.load(in_ptr38 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp199 = tmp197 * tmp198
    tmp200 = tl.load(in_ptr39 + ((-224) + x0), tmp183, eviction_policy='evict_last', other=0.0)
    tmp201 = tmp199 + tmp200
    tmp202 = 0.0
    tmp203 = triton_helpers.maximum(tmp201, tmp202)
    tmp204 = 6.0
    tmp205 = triton_helpers.minimum(tmp203, tmp204)
    tmp206 = tl.full(tmp205.shape, 0.0, tmp205.dtype)
    tmp207 = tl.where(tmp183, tmp205, tmp206)
    tmp208 = tl.where(tmp160, tmp182, tmp207)
    tmp209 = tl.where(tmp134, tmp156, tmp208)
    tmp210 = tl.where(tmp108, tmp130, tmp209)
    tmp211 = tl.where(tmp82, tmp104, tmp210)
    tmp212 = tl.where(tmp56, tmp78, tmp211)
    tmp213 = tl.where(tmp30, tmp52, tmp212)
    tmp214 = tl.where(tmp4, tmp26, tmp213)
    tl.store(out_ptr0 + (x7), tmp214, None)
