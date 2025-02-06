
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_log_mv_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 50, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_log_mv_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (4 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr2 + (1))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (8 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (2))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp29 = tl.load(in_ptr2 + (2))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp33 = tl.load(in_ptr0 + (12 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (3))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp40 = tl.load(in_ptr2 + (3))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp44 = tl.load(in_ptr3 + (16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp45 = tl.load(in_ptr4 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp51 = tl.load(in_ptr5 + (0))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp54 = tl.load(in_ptr6 + (0))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp57 = tl.load(in_ptr3 + (4 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp62 = tl.load(in_ptr5 + (1))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp65 = tl.load(in_ptr6 + (1))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp69 = tl.load(in_ptr3 + (8 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp74 = tl.load(in_ptr5 + (2))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp77 = tl.load(in_ptr6 + (2))
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK])
    tmp81 = tl.load(in_ptr3 + (12 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp86 = tl.load(in_ptr5 + (3))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp89 = tl.load(in_ptr6 + (3))
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK])
    tmp93 = tl.load(in_ptr7 + (16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr8 + (0))
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK])
    tmp100 = tl.load(in_ptr9 + (0))
    tmp101 = tl.broadcast_to(tmp100, [XBLOCK])
    tmp103 = tl.load(in_ptr7 + (4 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr8 + (1))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp110 = tl.load(in_ptr9 + (1))
    tmp111 = tl.broadcast_to(tmp110, [XBLOCK])
    tmp114 = tl.load(in_ptr7 + (8 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr8 + (2))
    tmp119 = tl.broadcast_to(tmp118, [XBLOCK])
    tmp121 = tl.load(in_ptr9 + (2))
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK])
    tmp125 = tl.load(in_ptr7 + (12 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr8 + (3))
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK])
    tmp132 = tl.load(in_ptr9 + (3))
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK])
    tmp136 = tl.load(in_ptr10 + (16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp137 = tl.load(in_ptr11 + (0))
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK])
    tmp143 = tl.load(in_ptr12 + (0))
    tmp144 = tl.broadcast_to(tmp143, [XBLOCK])
    tmp146 = tl.load(in_ptr13 + (0))
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK])
    tmp149 = tl.load(in_ptr10 + (4 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp154 = tl.load(in_ptr12 + (1))
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK])
    tmp157 = tl.load(in_ptr13 + (1))
    tmp158 = tl.broadcast_to(tmp157, [XBLOCK])
    tmp161 = tl.load(in_ptr10 + (8 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp166 = tl.load(in_ptr12 + (2))
    tmp167 = tl.broadcast_to(tmp166, [XBLOCK])
    tmp169 = tl.load(in_ptr13 + (2))
    tmp170 = tl.broadcast_to(tmp169, [XBLOCK])
    tmp173 = tl.load(in_ptr10 + (12 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp178 = tl.load(in_ptr12 + (3))
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK])
    tmp181 = tl.load(in_ptr13 + (3))
    tmp182 = tl.broadcast_to(tmp181, [XBLOCK])
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = tmp1 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp7 = tmp4 * tmp6
    tmp10 = tmp7 * tmp9
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp3 - tmp13
    tmp17 = tmp14 * tmp16
    tmp20 = tmp17 * tmp19
    tmp21 = tmp10 + tmp20
    tmp23 = libdevice.tanh(tmp22)
    tmp24 = tmp23 * tmp23
    tmp25 = tmp3 - tmp24
    tmp28 = tmp25 * tmp27
    tmp31 = tmp28 * tmp30
    tmp32 = tmp21 + tmp31
    tmp34 = libdevice.tanh(tmp33)
    tmp35 = tmp34 * tmp34
    tmp36 = tmp3 - tmp35
    tmp39 = tmp36 * tmp38
    tmp42 = tmp39 * tmp41
    tmp43 = tmp32 + tmp42
    tmp47 = tmp44 + tmp46
    tmp48 = libdevice.tanh(tmp47)
    tmp49 = tmp48 * tmp48
    tmp50 = tmp3 - tmp49
    tmp53 = tmp50 * tmp52
    tmp56 = tmp53 * tmp55
    tmp58 = tmp57 + tmp46
    tmp59 = libdevice.tanh(tmp58)
    tmp60 = tmp59 * tmp59
    tmp61 = tmp3 - tmp60
    tmp64 = tmp61 * tmp63
    tmp67 = tmp64 * tmp66
    tmp68 = tmp56 + tmp67
    tmp70 = tmp69 + tmp46
    tmp71 = libdevice.tanh(tmp70)
    tmp72 = tmp71 * tmp71
    tmp73 = tmp3 - tmp72
    tmp76 = tmp73 * tmp75
    tmp79 = tmp76 * tmp78
    tmp80 = tmp68 + tmp79
    tmp82 = tmp81 + tmp46
    tmp83 = libdevice.tanh(tmp82)
    tmp84 = tmp83 * tmp83
    tmp85 = tmp3 - tmp84
    tmp88 = tmp85 * tmp87
    tmp91 = tmp88 * tmp90
    tmp92 = tmp80 + tmp91
    tmp94 = libdevice.tanh(tmp93)
    tmp95 = tmp94 * tmp94
    tmp96 = tmp3 - tmp95
    tmp99 = tmp96 * tmp98
    tmp102 = tmp99 * tmp101
    tmp104 = libdevice.tanh(tmp103)
    tmp105 = tmp104 * tmp104
    tmp106 = tmp3 - tmp105
    tmp109 = tmp106 * tmp108
    tmp112 = tmp109 * tmp111
    tmp113 = tmp102 + tmp112
    tmp115 = libdevice.tanh(tmp114)
    tmp116 = tmp115 * tmp115
    tmp117 = tmp3 - tmp116
    tmp120 = tmp117 * tmp119
    tmp123 = tmp120 * tmp122
    tmp124 = tmp113 + tmp123
    tmp126 = libdevice.tanh(tmp125)
    tmp127 = tmp126 * tmp126
    tmp128 = tmp3 - tmp127
    tmp131 = tmp128 * tmp130
    tmp134 = tmp131 * tmp133
    tmp135 = tmp124 + tmp134
    tmp139 = tmp136 + tmp138
    tmp140 = libdevice.tanh(tmp139)
    tmp141 = tmp140 * tmp140
    tmp142 = tmp3 - tmp141
    tmp145 = tmp142 * tmp144
    tmp148 = tmp145 * tmp147
    tmp150 = tmp149 + tmp138
    tmp151 = libdevice.tanh(tmp150)
    tmp152 = tmp151 * tmp151
    tmp153 = tmp3 - tmp152
    tmp156 = tmp153 * tmp155
    tmp159 = tmp156 * tmp158
    tmp160 = tmp148 + tmp159
    tmp162 = tmp161 + tmp138
    tmp163 = libdevice.tanh(tmp162)
    tmp164 = tmp163 * tmp163
    tmp165 = tmp3 - tmp164
    tmp168 = tmp165 * tmp167
    tmp171 = tmp168 * tmp170
    tmp172 = tmp160 + tmp171
    tmp174 = tmp173 + tmp138
    tmp175 = libdevice.tanh(tmp174)
    tmp176 = tmp175 * tmp175
    tmp177 = tmp3 - tmp176
    tmp180 = tmp177 * tmp179
    tmp183 = tmp180 * tmp182
    tmp184 = tmp172 + tmp183
    tmp185 = tmp43 + tmp3
    tmp186 = tl_math.abs(tmp185)
    tmp187 = 1e-15
    tmp188 = tmp186 + tmp187
    tmp189 = tl_math.log(tmp188)
    tmp190 = 0.0
    tmp191 = tmp189 + tmp190
    tmp192 = tmp92 + tmp3
    tmp193 = tl_math.abs(tmp192)
    tmp194 = tmp193 + tmp187
    tmp195 = tl_math.log(tmp194)
    tmp196 = tmp191 + tmp195
    tmp197 = tmp135 + tmp3
    tmp198 = tl_math.abs(tmp197)
    tmp199 = tmp198 + tmp187
    tmp200 = tl_math.log(tmp199)
    tmp201 = tmp196 + tmp200
    tmp202 = tmp184 + tmp3
    tmp203 = tl_math.abs(tmp202)
    tmp204 = tmp203 + tmp187
    tmp205 = tl_math.log(tmp204)
    tmp206 = tmp201 + tmp205
    tl.store(in_out_ptr0 + (x0), tmp206, xmask)
