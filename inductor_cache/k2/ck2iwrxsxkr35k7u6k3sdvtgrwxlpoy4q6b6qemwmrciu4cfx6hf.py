
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr4': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_clamp_mean_mul_sub_2', 'mutated_arg_names': ['in_out_ptr4'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_clamp_mean_mul_sub_2(in_out_ptr4, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    r3 = rindex // 4
    r2 = (rindex % 4)
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6 - tmp6
    tmp8 = 0.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 - tmp10
    tmp12 = tmp11 * tmp8
    tmp13 = tmp10 + tmp12
    tmp14 = tmp0 + tmp13
    tmp15 = r3
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 0.3333333333333333
    tmp18 = tmp16 * tmp17
    tmp19 = triton_helpers.maximum(tmp18, tmp8)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tl.full([1, 1], 1, tl.int64)
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.minimum(tmp22, tmp21)
    tmp24 = r2
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp17
    tmp27 = triton_helpers.maximum(tmp26, tmp8)
    tmp28 = tmp27.to(tl.int32)
    tmp29 = tmp28 + tmp21
    tmp30 = triton_helpers.minimum(tmp29, tmp21)
    tmp31 = tl.load(in_ptr0 + (2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (1 + 2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp33 = tmp32 + tmp31
    tmp34 = tl.load(in_ptr0 + (4 + 2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp35 = tmp34 + tmp33
    tmp36 = tl.load(in_ptr0 + (5 + 2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp37 = tmp36 + tmp35
    tmp38 = 0.25
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr0 + (2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (1 + 2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp42 = tmp41 + tmp40
    tmp43 = tl.load(in_ptr0 + (4 + 2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp44 = tmp43 + tmp42
    tmp45 = tl.load(in_ptr0 + (5 + 2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp46 = tmp45 + tmp44
    tmp47 = tmp46 * tmp38
    tmp48 = tmp39 - tmp47
    tmp49 = tl.load(in_ptr0 + (2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (1 + 2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp51 = tmp50 + tmp49
    tmp52 = tl.load(in_ptr0 + (4 + 2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp53 = tmp52 + tmp51
    tmp54 = tl.load(in_ptr0 + (5 + 2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp55 = tmp54 + tmp53
    tmp56 = tmp55 * tmp38
    tmp57 = tl.load(in_ptr0 + (2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (1 + 2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp59 = tmp58 + tmp57
    tmp60 = tl.load(in_ptr0 + (4 + 2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp61 = tmp60 + tmp59
    tmp62 = tl.load(in_ptr0 + (5 + 2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp63 = tmp62 + tmp61
    tmp64 = tmp63 * tmp38
    tmp65 = tmp56 - tmp64
    tmp66 = tmp28.to(tl.float32)
    tmp67 = tmp27 - tmp66
    tmp68 = triton_helpers.maximum(tmp67, tmp8)
    tmp69 = 1.0
    tmp70 = triton_helpers.minimum(tmp68, tmp69)
    tmp71 = tmp48 * tmp70
    tmp72 = tmp47 + tmp71
    tmp73 = tmp65 * tmp70
    tmp74 = tmp64 + tmp73
    tmp75 = 0.6666666666666666
    tmp76 = tmp16 * tmp75
    tmp77 = triton_helpers.maximum(tmp76, tmp8)
    tmp78 = tmp77.to(tl.int32)
    tmp79 = tmp78 + tmp21
    tmp80 = tl.full([1, 1], 2, tl.int64)
    tmp81 = triton_helpers.minimum(tmp79, tmp80)
    tmp82 = tmp25 * tmp75
    tmp83 = triton_helpers.maximum(tmp82, tmp8)
    tmp84 = tmp83.to(tl.int32)
    tmp85 = tl.load(in_ptr1 + (tmp84 + 3*tmp81 + 9*x0), xmask, eviction_policy='evict_last')
    tmp86 = tmp84 + tmp21
    tmp87 = triton_helpers.minimum(tmp86, tmp80)
    tmp88 = tl.load(in_ptr1 + (tmp87 + 3*tmp81 + 9*x0), xmask, eviction_policy='evict_last')
    tmp89 = tmp88 - tmp85
    tmp90 = tmp84.to(tl.float32)
    tmp91 = tmp83 - tmp90
    tmp92 = triton_helpers.maximum(tmp91, tmp8)
    tmp93 = triton_helpers.minimum(tmp92, tmp69)
    tmp94 = tmp89 * tmp93
    tmp95 = tmp85 + tmp94
    tmp96 = tl.load(in_ptr1 + (tmp84 + 3*tmp78 + 9*x0), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr1 + (tmp87 + 3*tmp78 + 9*x0), xmask, eviction_policy='evict_last')
    tmp98 = tmp97 - tmp96
    tmp99 = tmp98 * tmp93
    tmp100 = tmp96 + tmp99
    tmp101 = tmp95 - tmp100
    tmp102 = tmp78.to(tl.float32)
    tmp103 = tmp77 - tmp102
    tmp104 = triton_helpers.maximum(tmp103, tmp8)
    tmp105 = triton_helpers.minimum(tmp104, tmp69)
    tmp106 = tmp101 * tmp105
    tmp107 = tmp100 + tmp106
    tmp108 = 1.6666666666666667
    tmp109 = tmp16 * tmp108
    tmp110 = triton_helpers.maximum(tmp109, tmp8)
    tmp111 = tmp110.to(tl.int32)
    tmp112 = tmp111 + tmp21
    tmp113 = tl.full([1, 1], 5, tl.int64)
    tmp114 = triton_helpers.minimum(tmp112, tmp113)
    tmp115 = tmp25 * tmp108
    tmp116 = triton_helpers.maximum(tmp115, tmp8)
    tmp117 = tmp116.to(tl.int32)
    tmp118 = tl.load(in_ptr2 + (tmp117 + 6*tmp114 + 36*x0), xmask, eviction_policy='evict_last')
    tmp119 = tmp117 + tmp21
    tmp120 = triton_helpers.minimum(tmp119, tmp113)
    tmp121 = tl.load(in_ptr2 + (tmp120 + 6*tmp114 + 36*x0), xmask, eviction_policy='evict_last')
    tmp122 = tmp121 - tmp118
    tmp123 = tmp117.to(tl.float32)
    tmp124 = tmp116 - tmp123
    tmp125 = triton_helpers.maximum(tmp124, tmp8)
    tmp126 = triton_helpers.minimum(tmp125, tmp69)
    tmp127 = tmp122 * tmp126
    tmp128 = tmp118 + tmp127
    tmp129 = tl.load(in_ptr2 + (tmp117 + 6*tmp111 + 36*x0), xmask, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr2 + (tmp120 + 6*tmp111 + 36*x0), xmask, eviction_policy='evict_last')
    tmp131 = tmp130 - tmp129
    tmp132 = tmp131 * tmp126
    tmp133 = tmp129 + tmp132
    tmp134 = tmp128 - tmp133
    tmp135 = tmp111.to(tl.float32)
    tmp136 = tmp110 - tmp135
    tmp137 = triton_helpers.maximum(tmp136, tmp8)
    tmp138 = triton_helpers.minimum(tmp137, tmp69)
    tmp139 = tmp134 * tmp138
    tmp140 = tmp133 + tmp139
    tmp141 = tmp72 - tmp74
    tmp142 = tmp20.to(tl.float32)
    tmp143 = tmp19 - tmp142
    tmp144 = triton_helpers.maximum(tmp143, tmp8)
    tmp145 = triton_helpers.minimum(tmp144, tmp69)
    tmp146 = tmp141 * tmp145
    tmp147 = tmp74 + tmp146
    tmp148 = tmp14 + tmp147
    tmp149 = tmp148 + tmp107
    tmp150 = tmp149 + tmp140
    tl.store(in_out_ptr4 + (r1 + 16*x0), tmp150, xmask)
