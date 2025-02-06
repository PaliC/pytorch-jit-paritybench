
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_reflection_pad2d_rsub_sub_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 48, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_avg_pool2d_clamp_div_mul_pow_reflection_pad2d_rsub_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 5)
    x1 = ((xindex // 5) % 5)
    x2 = xindex // 25
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8*x1 + 64*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (1 + x0 + 8*x1 + 64*x2), xmask)
    tmp3 = tl.load(in_ptr0 + (2 + x0 + 8*x1 + 64*x2), xmask)
    tmp5 = tl.load(in_ptr0 + (3 + x0 + 8*x1 + 64*x2), xmask)
    tmp7 = tl.load(in_ptr0 + (8 + x0 + 8*x1 + 64*x2), xmask)
    tmp9 = tl.load(in_ptr0 + (9 + x0 + 8*x1 + 64*x2), xmask)
    tmp11 = tl.load(in_ptr0 + (10 + x0 + 8*x1 + 64*x2), xmask)
    tmp13 = tl.load(in_ptr0 + (11 + x0 + 8*x1 + 64*x2), xmask)
    tmp15 = tl.load(in_ptr0 + (16 + x0 + 8*x1 + 64*x2), xmask)
    tmp17 = tl.load(in_ptr0 + (17 + x0 + 8*x1 + 64*x2), xmask)
    tmp19 = tl.load(in_ptr0 + (18 + x0 + 8*x1 + 64*x2), xmask)
    tmp21 = tl.load(in_ptr0 + (19 + x0 + 8*x1 + 64*x2), xmask)
    tmp23 = tl.load(in_ptr0 + (24 + x0 + 8*x1 + 64*x2), xmask)
    tmp25 = tl.load(in_ptr0 + (25 + x0 + 8*x1 + 64*x2), xmask)
    tmp27 = tl.load(in_ptr0 + (26 + x0 + 8*x1 + 64*x2), xmask)
    tmp29 = tl.load(in_ptr0 + (27 + x0 + 8*x1 + 64*x2), xmask)
    tmp33 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask)
    tmp38 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask)
    tmp40 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask)
    tmp46 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask)
    tmp48 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask)
    tmp54 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask)
    tmp56 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask)
    tmp62 = tl.load(in_ptr1 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask)
    tmp97 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp100 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask)
    tmp102 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-2) + x1))) + 16*x2), xmask)
    tmp104 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask)
    tmp110 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-3) + tl_math.abs((-1) + x1))) + 16*x2), xmask)
    tmp112 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask)
    tmp118 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-3) + x1)) + 16*x2), xmask)
    tmp120 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-2) + x0))) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + tl_math.abs((-1) + x0))) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-3) + x0)) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask)
    tmp126 = tl.load(in_ptr2 + (15 + ((-1)*tl_math.abs((-2) + x0)) + ((-4)*tl_math.abs((-2) + x1)) + 16*x2), xmask)
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp8 = tmp7 + tmp6
    tmp10 = tmp9 + tmp8
    tmp12 = tmp11 + tmp10
    tmp14 = tmp13 + tmp12
    tmp16 = tmp15 + tmp14
    tmp18 = tmp17 + tmp16
    tmp20 = tmp19 + tmp18
    tmp22 = tmp21 + tmp20
    tmp24 = tmp23 + tmp22
    tmp26 = tmp25 + tmp24
    tmp28 = tmp27 + tmp26
    tmp30 = tmp29 + tmp28
    tmp31 = 0.0625
    tmp32 = tmp30 * tmp31
    tmp35 = tmp34 + tmp33
    tmp37 = tmp36 + tmp35
    tmp39 = tmp38 + tmp37
    tmp41 = tmp40 + tmp39
    tmp43 = tmp42 + tmp41
    tmp45 = tmp44 + tmp43
    tmp47 = tmp46 + tmp45
    tmp49 = tmp48 + tmp47
    tmp51 = tmp50 + tmp49
    tmp53 = tmp52 + tmp51
    tmp55 = tmp54 + tmp53
    tmp57 = tmp56 + tmp55
    tmp59 = tmp58 + tmp57
    tmp61 = tmp60 + tmp59
    tmp63 = tmp62 + tmp61
    tmp64 = tmp63 * tmp31
    tmp65 = tmp33 * tmp33
    tmp66 = tmp34 * tmp34
    tmp67 = tmp66 + tmp65
    tmp68 = tmp36 * tmp36
    tmp69 = tmp68 + tmp67
    tmp70 = tmp38 * tmp38
    tmp71 = tmp70 + tmp69
    tmp72 = tmp40 * tmp40
    tmp73 = tmp72 + tmp71
    tmp74 = tmp42 * tmp42
    tmp75 = tmp74 + tmp73
    tmp76 = tmp44 * tmp44
    tmp77 = tmp76 + tmp75
    tmp78 = tmp46 * tmp46
    tmp79 = tmp78 + tmp77
    tmp80 = tmp48 * tmp48
    tmp81 = tmp80 + tmp79
    tmp82 = tmp50 * tmp50
    tmp83 = tmp82 + tmp81
    tmp84 = tmp52 * tmp52
    tmp85 = tmp84 + tmp83
    tmp86 = tmp54 * tmp54
    tmp87 = tmp86 + tmp85
    tmp88 = tmp56 * tmp56
    tmp89 = tmp88 + tmp87
    tmp90 = tmp58 * tmp58
    tmp91 = tmp90 + tmp89
    tmp92 = tmp60 * tmp60
    tmp93 = tmp92 + tmp91
    tmp94 = tmp62 * tmp62
    tmp95 = tmp94 + tmp93
    tmp96 = tmp95 * tmp31
    tmp99 = tmp98 + tmp97
    tmp101 = tmp100 + tmp99
    tmp103 = tmp102 + tmp101
    tmp105 = tmp104 + tmp103
    tmp107 = tmp106 + tmp105
    tmp109 = tmp108 + tmp107
    tmp111 = tmp110 + tmp109
    tmp113 = tmp112 + tmp111
    tmp115 = tmp114 + tmp113
    tmp117 = tmp116 + tmp115
    tmp119 = tmp118 + tmp117
    tmp121 = tmp120 + tmp119
    tmp123 = tmp122 + tmp121
    tmp125 = tmp124 + tmp123
    tmp127 = tmp126 + tmp125
    tmp128 = tmp127 * tmp31
    tmp129 = tmp97 * tmp97
    tmp130 = tmp98 * tmp98
    tmp131 = tmp130 + tmp129
    tmp132 = tmp100 * tmp100
    tmp133 = tmp132 + tmp131
    tmp134 = tmp102 * tmp102
    tmp135 = tmp134 + tmp133
    tmp136 = tmp104 * tmp104
    tmp137 = tmp136 + tmp135
    tmp138 = tmp106 * tmp106
    tmp139 = tmp138 + tmp137
    tmp140 = tmp108 * tmp108
    tmp141 = tmp140 + tmp139
    tmp142 = tmp110 * tmp110
    tmp143 = tmp142 + tmp141
    tmp144 = tmp112 * tmp112
    tmp145 = tmp144 + tmp143
    tmp146 = tmp114 * tmp114
    tmp147 = tmp146 + tmp145
    tmp148 = tmp116 * tmp116
    tmp149 = tmp148 + tmp147
    tmp150 = tmp118 * tmp118
    tmp151 = tmp150 + tmp149
    tmp152 = tmp120 * tmp120
    tmp153 = tmp152 + tmp151
    tmp154 = tmp122 * tmp122
    tmp155 = tmp154 + tmp153
    tmp156 = tmp124 * tmp124
    tmp157 = tmp156 + tmp155
    tmp158 = tmp126 * tmp126
    tmp159 = tmp158 + tmp157
    tmp160 = tmp159 * tmp31
    tmp161 = 2.0
    tmp162 = tmp64 * tmp161
    tmp163 = tmp162 * tmp128
    tmp164 = 0.0001
    tmp165 = tmp163 + tmp164
    tmp166 = tmp64 * tmp128
    tmp167 = tmp32 - tmp166
    tmp168 = tmp167 * tmp161
    tmp169 = 0.0009
    tmp170 = tmp168 + tmp169
    tmp171 = tmp165 * tmp170
    tmp172 = tmp64 * tmp64
    tmp173 = tmp128 * tmp128
    tmp174 = tmp172 + tmp173
    tmp175 = tmp174 + tmp164
    tmp176 = tmp96 - tmp172
    tmp177 = tmp160 - tmp173
    tmp178 = tmp176 + tmp177
    tmp179 = tmp178 + tmp169
    tmp180 = tmp175 * tmp179
    tmp181 = tmp171 / tmp180
    tmp182 = 1.0
    tmp183 = tmp182 - tmp181
    tmp184 = 0.5
    tmp185 = tmp183 * tmp184
    tmp186 = 0.0
    tmp187 = triton_helpers.maximum(tmp185, tmp186)
    tmp188 = triton_helpers.minimum(tmp187, tmp182)
    tl.store(in_out_ptr0 + (x3), tmp188, xmask)
