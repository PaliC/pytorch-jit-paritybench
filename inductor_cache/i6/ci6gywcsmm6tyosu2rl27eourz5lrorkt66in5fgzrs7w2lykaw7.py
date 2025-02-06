
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 4), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clamp_div_index_max_mul_neg_pow_reciprocal_rsub_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clamp_div_index_max_mul_neg_pow_reciprocal_rsub_sub_sum_0(in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 16)
    r1 = rindex // 16
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 64*r1), None)
    tmp1 = tl.load(in_ptr0 + (16 + r0 + 64*r1), None)
    tmp3 = tl.load(in_ptr0 + (32 + r0 + 64*r1), None)
    tmp5 = tl.load(in_ptr0 + (48 + r0 + 64*r1), None)
    tmp127 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp0 - tmp6
    tmp8 = -3.0
    tmp9 = tmp7 * tmp8
    tmp10 = 1.0
    tmp11 = tmp9 + tmp10
    tmp12 = 0.0
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = -0.3333333333333333
    tmp15 = libdevice.pow(tmp13, tmp14)
    tmp16 = tmp1 - tmp6
    tmp17 = tmp16 * tmp8
    tmp18 = tmp17 + tmp10
    tmp19 = triton_helpers.maximum(tmp18, tmp12)
    tmp20 = libdevice.pow(tmp19, tmp14)
    tmp21 = tmp15 + tmp20
    tmp22 = tmp3 - tmp6
    tmp23 = tmp22 * tmp8
    tmp24 = tmp23 + tmp10
    tmp25 = triton_helpers.maximum(tmp24, tmp12)
    tmp26 = libdevice.pow(tmp25, tmp14)
    tmp27 = tmp21 + tmp26
    tmp28 = tmp5 - tmp6
    tmp29 = tmp28 * tmp8
    tmp30 = tmp29 + tmp10
    tmp31 = triton_helpers.maximum(tmp30, tmp12)
    tmp32 = libdevice.pow(tmp31, tmp14)
    tmp33 = tmp27 + tmp32
    tmp34 = tl.full([1, 1], 1, tl.int32)
    tmp35 = tmp34 / tmp33
    tmp36 = tmp35 * tmp35
    tmp37 = tmp36 * tmp35
    tmp38 = tmp37 * tmp7
    tmp39 = tmp38 * tmp8
    tmp40 = tmp39 + tmp10
    tmp41 = triton_helpers.maximum(tmp40, tmp12)
    tmp42 = libdevice.pow(tmp41, tmp14)
    tmp43 = tmp37 * tmp16
    tmp44 = tmp43 * tmp8
    tmp45 = tmp44 + tmp10
    tmp46 = triton_helpers.maximum(tmp45, tmp12)
    tmp47 = libdevice.pow(tmp46, tmp14)
    tmp48 = tmp42 + tmp47
    tmp49 = tmp37 * tmp22
    tmp50 = tmp49 * tmp8
    tmp51 = tmp50 + tmp10
    tmp52 = triton_helpers.maximum(tmp51, tmp12)
    tmp53 = libdevice.pow(tmp52, tmp14)
    tmp54 = tmp48 + tmp53
    tmp55 = tmp37 * tmp28
    tmp56 = tmp55 * tmp8
    tmp57 = tmp56 + tmp10
    tmp58 = triton_helpers.maximum(tmp57, tmp12)
    tmp59 = libdevice.pow(tmp58, tmp14)
    tmp60 = tmp54 + tmp59
    tmp61 = tmp34 / tmp60
    tmp62 = tmp61 * tmp61
    tmp63 = tmp62 * tmp61
    tmp64 = tmp63 * tmp7
    tmp65 = tmp64 * tmp8
    tmp66 = tmp65 + tmp10
    tmp67 = triton_helpers.maximum(tmp66, tmp12)
    tmp68 = libdevice.pow(tmp67, tmp14)
    tmp69 = tmp63 * tmp16
    tmp70 = tmp69 * tmp8
    tmp71 = tmp70 + tmp10
    tmp72 = triton_helpers.maximum(tmp71, tmp12)
    tmp73 = libdevice.pow(tmp72, tmp14)
    tmp74 = tmp68 + tmp73
    tmp75 = tmp63 * tmp22
    tmp76 = tmp75 * tmp8
    tmp77 = tmp76 + tmp10
    tmp78 = triton_helpers.maximum(tmp77, tmp12)
    tmp79 = libdevice.pow(tmp78, tmp14)
    tmp80 = tmp74 + tmp79
    tmp81 = tmp63 * tmp28
    tmp82 = tmp81 * tmp8
    tmp83 = tmp82 + tmp10
    tmp84 = triton_helpers.maximum(tmp83, tmp12)
    tmp85 = libdevice.pow(tmp84, tmp14)
    tmp86 = tmp80 + tmp85
    tmp87 = tmp34 / tmp86
    tmp88 = tmp87 * tmp10
    tmp89 = tmp34 / tmp88
    tmp90 = tmp89 * tmp89
    tmp91 = tmp90 * tmp89
    tmp92 = tmp91 - tmp10
    tmp93 = tmp92 * tmp14
    tmp94 = -tmp93
    tmp95 = tmp94 + tmp6
    tmp96 = tmp0 - tmp95
    tmp97 = tmp96 * tmp8
    tmp98 = tmp97 + tmp10
    tmp99 = triton_helpers.maximum(tmp98, tmp12)
    tmp100 = libdevice.pow(tmp99, tmp14)
    tmp101 = tmp34 / tmp100
    tmp102 = tmp101 * tmp101
    tmp103 = tmp1 - tmp95
    tmp104 = tmp103 * tmp8
    tmp105 = tmp104 + tmp10
    tmp106 = triton_helpers.maximum(tmp105, tmp12)
    tmp107 = libdevice.pow(tmp106, tmp14)
    tmp108 = tmp34 / tmp107
    tmp109 = tmp108 * tmp108
    tmp110 = tmp102 + tmp109
    tmp111 = tmp3 - tmp95
    tmp112 = tmp111 * tmp8
    tmp113 = tmp112 + tmp10
    tmp114 = triton_helpers.maximum(tmp113, tmp12)
    tmp115 = libdevice.pow(tmp114, tmp14)
    tmp116 = tmp34 / tmp115
    tmp117 = tmp116 * tmp116
    tmp118 = tmp110 + tmp117
    tmp119 = tmp5 - tmp95
    tmp120 = tmp119 * tmp8
    tmp121 = tmp120 + tmp10
    tmp122 = triton_helpers.maximum(tmp121, tmp12)
    tmp123 = libdevice.pow(tmp122, tmp14)
    tmp124 = tmp34 / tmp123
    tmp125 = tmp124 * tmp124
    tmp126 = tmp118 + tmp125
    tmp128 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
    tmp129 = tmp127 + tmp128
    tmp130 = tmp127 < 0
    tmp131 = tl.where(tmp130, tmp129, tmp127)
    tl.device_assert((0 <= tmp131) & (tmp131 < 4), "index out of bounds: 0 <= tmp131 < 4")
    tmp133 = tl.load(in_ptr0 + (r0 + 16*tmp131 + 64*r1), None)
    tmp134 = tmp133 - tmp95
    tmp135 = tmp134 * tmp8
    tmp136 = tmp135 + tmp10
    tmp137 = triton_helpers.maximum(tmp136, tmp12)
    tmp138 = libdevice.pow(tmp137, tmp14)
    tmp139 = tmp34 / tmp138
    tmp140 = tmp139 * tmp139
    tmp141 = tmp140 * tmp139
    tmp142 = tmp141 - tmp10
    tmp143 = tmp142 * tmp14
    tmp144 = -tmp143
    tmp145 = tmp10 - tmp126
    tmp146 = -0.5
    tmp147 = tmp145 * tmp146
    tmp148 = tmp144 - tmp147
    tmp149 = 0.25
    tmp150 = tmp148 * tmp149
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp153 = tl.sum(tmp151, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp153, None)
