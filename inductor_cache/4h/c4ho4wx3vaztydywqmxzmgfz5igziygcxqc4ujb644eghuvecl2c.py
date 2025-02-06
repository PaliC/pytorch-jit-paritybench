
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_rsub_sub_sum_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 16, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_rsub_sub_sum_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (64 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (65 + 4*r0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (66 + 4*r0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (67 + 4*r0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (128 + 4*r0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (129 + 4*r0), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (130 + 4*r0), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (131 + 4*r0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr0 + (192 + 4*r0), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (193 + 4*r0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr0 + (194 + 4*r0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr0 + (195 + 4*r0), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr2 + (4*r0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr2 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr2 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr2 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr3 + (4*r0), None, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr3 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp109 = tl.load(in_ptr3 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr3 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp148 = tl.load(in_ptr4 + (4*r0), None, eviction_policy='evict_last')
    tmp150 = tl.load(in_ptr4 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr4 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp156 = tl.load(in_ptr4 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp19 = tmp18 * tmp1
    tmp21 = tmp20 * tmp4
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23 * tmp8
    tmp25 = tmp22 + tmp24
    tmp27 = tmp26 * tmp12
    tmp28 = tmp25 + tmp27
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp31 = tl.sum(tmp29, 1)[:, None]
    tmp33 = tmp32 * tmp1
    tmp35 = tmp34 * tmp4
    tmp36 = tmp33 + tmp35
    tmp38 = tmp37 * tmp8
    tmp39 = tmp36 + tmp38
    tmp41 = tmp40 * tmp12
    tmp42 = tmp39 + tmp41
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp45 = tl.sum(tmp43, 1)[:, None]
    tmp47 = tmp46 * tmp1
    tmp49 = tmp48 * tmp4
    tmp50 = tmp47 + tmp49
    tmp52 = tmp51 * tmp8
    tmp53 = tmp50 + tmp52
    tmp55 = tmp54 * tmp12
    tmp56 = tmp53 + tmp55
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.sum(tmp57, 1)[:, None]
    tmp61 = tmp0 * tmp60
    tmp63 = tmp3 * tmp62
    tmp64 = tmp61 + tmp63
    tmp66 = tmp7 * tmp65
    tmp67 = tmp64 + tmp66
    tmp69 = tmp11 * tmp68
    tmp70 = tmp67 + tmp69
    tmp71 = tl.broadcast_to(tmp70, [XBLOCK, RBLOCK])
    tmp73 = tl.sum(tmp71, 1)[:, None]
    tmp74 = tmp18 * tmp60
    tmp75 = tmp20 * tmp62
    tmp76 = tmp74 + tmp75
    tmp77 = tmp23 * tmp65
    tmp78 = tmp76 + tmp77
    tmp79 = tmp26 * tmp68
    tmp80 = tmp78 + tmp79
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp83 = tl.sum(tmp81, 1)[:, None]
    tmp84 = tmp32 * tmp60
    tmp85 = tmp34 * tmp62
    tmp86 = tmp84 + tmp85
    tmp87 = tmp37 * tmp65
    tmp88 = tmp86 + tmp87
    tmp89 = tmp40 * tmp68
    tmp90 = tmp88 + tmp89
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK, RBLOCK])
    tmp93 = tl.sum(tmp91, 1)[:, None]
    tmp94 = tmp46 * tmp60
    tmp95 = tmp48 * tmp62
    tmp96 = tmp94 + tmp95
    tmp97 = tmp51 * tmp65
    tmp98 = tmp96 + tmp97
    tmp99 = tmp54 * tmp68
    tmp100 = tmp98 + tmp99
    tmp101 = tl.broadcast_to(tmp100, [XBLOCK, RBLOCK])
    tmp103 = tl.sum(tmp101, 1)[:, None]
    tmp105 = tmp0 * tmp104
    tmp107 = tmp3 * tmp106
    tmp108 = tmp105 + tmp107
    tmp110 = tmp7 * tmp109
    tmp111 = tmp108 + tmp110
    tmp113 = tmp11 * tmp112
    tmp114 = tmp111 + tmp113
    tmp115 = tl.broadcast_to(tmp114, [XBLOCK, RBLOCK])
    tmp117 = tl.sum(tmp115, 1)[:, None]
    tmp118 = tmp18 * tmp104
    tmp119 = tmp20 * tmp106
    tmp120 = tmp118 + tmp119
    tmp121 = tmp23 * tmp109
    tmp122 = tmp120 + tmp121
    tmp123 = tmp26 * tmp112
    tmp124 = tmp122 + tmp123
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp127 = tl.sum(tmp125, 1)[:, None]
    tmp128 = tmp32 * tmp104
    tmp129 = tmp34 * tmp106
    tmp130 = tmp128 + tmp129
    tmp131 = tmp37 * tmp109
    tmp132 = tmp130 + tmp131
    tmp133 = tmp40 * tmp112
    tmp134 = tmp132 + tmp133
    tmp135 = tl.broadcast_to(tmp134, [XBLOCK, RBLOCK])
    tmp137 = tl.sum(tmp135, 1)[:, None]
    tmp138 = tmp46 * tmp104
    tmp139 = tmp48 * tmp106
    tmp140 = tmp138 + tmp139
    tmp141 = tmp51 * tmp109
    tmp142 = tmp140 + tmp141
    tmp143 = tmp54 * tmp112
    tmp144 = tmp142 + tmp143
    tmp145 = tl.broadcast_to(tmp144, [XBLOCK, RBLOCK])
    tmp147 = tl.sum(tmp145, 1)[:, None]
    tmp149 = tmp0 * tmp148
    tmp151 = tmp3 * tmp150
    tmp152 = tmp149 + tmp151
    tmp154 = tmp7 * tmp153
    tmp155 = tmp152 + tmp154
    tmp157 = tmp11 * tmp156
    tmp158 = tmp155 + tmp157
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, RBLOCK])
    tmp161 = tl.sum(tmp159, 1)[:, None]
    tmp162 = tmp18 * tmp148
    tmp163 = tmp20 * tmp150
    tmp164 = tmp162 + tmp163
    tmp165 = tmp23 * tmp153
    tmp166 = tmp164 + tmp165
    tmp167 = tmp26 * tmp156
    tmp168 = tmp166 + tmp167
    tmp169 = tl.broadcast_to(tmp168, [XBLOCK, RBLOCK])
    tmp171 = tl.sum(tmp169, 1)[:, None]
    tmp172 = tmp32 * tmp148
    tmp173 = tmp34 * tmp150
    tmp174 = tmp172 + tmp173
    tmp175 = tmp37 * tmp153
    tmp176 = tmp174 + tmp175
    tmp177 = tmp40 * tmp156
    tmp178 = tmp176 + tmp177
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK, RBLOCK])
    tmp181 = tl.sum(tmp179, 1)[:, None]
    tmp182 = tmp46 * tmp148
    tmp183 = tmp48 * tmp150
    tmp184 = tmp182 + tmp183
    tmp185 = tmp51 * tmp153
    tmp186 = tmp184 + tmp185
    tmp187 = tmp54 * tmp156
    tmp188 = tmp186 + tmp187
    tmp189 = tl.broadcast_to(tmp188, [XBLOCK, RBLOCK])
    tmp191 = tl.sum(tmp189, 1)[:, None]
    tmp192 = 16.0
    tmp193 = tmp17 / tmp192
    tmp194 = 0.0
    tmp195 = tmp194 - tmp193
    tmp196 = tmp31 / tmp192
    tmp197 = tmp195 - tmp196
    tmp198 = tmp45 / tmp192
    tmp199 = tmp197 - tmp198
    tmp200 = tmp59 / tmp192
    tmp201 = tmp199 - tmp200
    tmp202 = tmp73 / tmp192
    tmp203 = tmp201 - tmp202
    tmp204 = tmp83 / tmp192
    tmp205 = tmp203 - tmp204
    tmp206 = tmp93 / tmp192
    tmp207 = tmp205 - tmp206
    tmp208 = tmp103 / tmp192
    tmp209 = tmp207 - tmp208
    tmp210 = tmp117 / tmp192
    tmp211 = tmp209 - tmp210
    tmp212 = tmp127 / tmp192
    tmp213 = tmp211 - tmp212
    tmp214 = tmp137 / tmp192
    tmp215 = tmp213 - tmp214
    tmp216 = tmp147 / tmp192
    tmp217 = tmp215 - tmp216
    tmp218 = tmp161 / tmp192
    tmp219 = tmp217 - tmp218
    tmp220 = tmp171 / tmp192
    tmp221 = tmp219 - tmp220
    tmp222 = tmp181 / tmp192
    tmp223 = tmp221 - tmp222
    tmp224 = tmp191 / tmp192
    tmp225 = tmp223 - tmp224
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp225, None)
