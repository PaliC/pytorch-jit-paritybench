
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mul_ne_pow_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mul_ne_pow_sum_view_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp153 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.int64)
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = tl.full([1], 3, tl.int64)
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp6 == tmp3
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp0 * tmp9
    tmp11 = 255.0
    tmp12 = tmp1 != tmp11
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp10 * tmp14
    tmp17 = tmp16.to(tl.int64)
    tmp18 = triton_helpers.maximum(tmp17, tmp3)
    tmp19 = triton_helpers.minimum(tmp18, tmp5)
    tmp20 = tmp19 == tmp3
    tmp21 = tmp20.to(tl.int64)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp0 * tmp22
    tmp24 = tmp16 != tmp11
    tmp25 = tmp24.to(tl.int64)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp23 * tmp26
    tmp28 = tmp15 + tmp27
    tmp30 = tmp29.to(tl.int64)
    tmp31 = triton_helpers.maximum(tmp30, tmp3)
    tmp32 = triton_helpers.minimum(tmp31, tmp5)
    tmp33 = tmp32 == tmp3
    tmp34 = tmp33.to(tl.int64)
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp0 * tmp35
    tmp37 = tmp29 != tmp11
    tmp38 = tmp37.to(tl.int64)
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp36 * tmp39
    tmp41 = tmp28 + tmp40
    tmp43 = tmp42.to(tl.int64)
    tmp44 = triton_helpers.maximum(tmp43, tmp3)
    tmp45 = triton_helpers.minimum(tmp44, tmp5)
    tmp46 = tmp45 == tmp3
    tmp47 = tmp46.to(tl.int64)
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp0 * tmp48
    tmp50 = tmp42 != tmp11
    tmp51 = tmp50.to(tl.int64)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp49 * tmp52
    tmp54 = tmp41 + tmp53
    tmp55 = tmp0 * tmp0
    tmp56 = tmp8 * tmp8
    tmp57 = tmp56.to(tl.float32)
    tmp58 = tmp55 + tmp57
    tmp59 = tmp21 * tmp21
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp55 + tmp60
    tmp62 = tmp58 + tmp61
    tmp63 = tmp34 * tmp34
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp55 + tmp64
    tmp66 = tmp62 + tmp65
    tmp67 = tmp47 * tmp47
    tmp68 = tmp67.to(tl.float32)
    tmp69 = tmp55 + tmp68
    tmp70 = tmp66 + tmp69
    tmp72 = tl.full([1], 1, tl.int64)
    tmp73 = tmp6 == tmp72
    tmp74 = tmp73.to(tl.int64)
    tmp75 = tmp74.to(tl.float32)
    tmp76 = tmp71 * tmp75
    tmp77 = tmp76 * tmp14
    tmp78 = tmp19 == tmp72
    tmp79 = tmp78.to(tl.int64)
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp71 * tmp80
    tmp82 = tmp81 * tmp26
    tmp83 = tmp77 + tmp82
    tmp84 = tmp32 == tmp72
    tmp85 = tmp84.to(tl.int64)
    tmp86 = tmp85.to(tl.float32)
    tmp87 = tmp71 * tmp86
    tmp88 = tmp87 * tmp39
    tmp89 = tmp83 + tmp88
    tmp90 = tmp45 == tmp72
    tmp91 = tmp90.to(tl.int64)
    tmp92 = tmp91.to(tl.float32)
    tmp93 = tmp71 * tmp92
    tmp94 = tmp93 * tmp52
    tmp95 = tmp89 + tmp94
    tmp96 = tmp71 * tmp71
    tmp97 = tmp74 * tmp74
    tmp98 = tmp97.to(tl.float32)
    tmp99 = tmp96 + tmp98
    tmp100 = tmp79 * tmp79
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tmp96 + tmp101
    tmp103 = tmp99 + tmp102
    tmp104 = tmp85 * tmp85
    tmp105 = tmp104.to(tl.float32)
    tmp106 = tmp96 + tmp105
    tmp107 = tmp103 + tmp106
    tmp108 = tmp91 * tmp91
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp96 + tmp109
    tmp111 = tmp107 + tmp110
    tmp113 = tl.full([1], 2, tl.int64)
    tmp114 = tmp6 == tmp113
    tmp115 = tmp114.to(tl.int64)
    tmp116 = tmp115.to(tl.float32)
    tmp117 = tmp112 * tmp116
    tmp118 = tmp117 * tmp14
    tmp119 = tmp19 == tmp113
    tmp120 = tmp119.to(tl.int64)
    tmp121 = tmp120.to(tl.float32)
    tmp122 = tmp112 * tmp121
    tmp123 = tmp122 * tmp26
    tmp124 = tmp118 + tmp123
    tmp125 = tmp32 == tmp113
    tmp126 = tmp125.to(tl.int64)
    tmp127 = tmp126.to(tl.float32)
    tmp128 = tmp112 * tmp127
    tmp129 = tmp128 * tmp39
    tmp130 = tmp124 + tmp129
    tmp131 = tmp45 == tmp113
    tmp132 = tmp131.to(tl.int64)
    tmp133 = tmp132.to(tl.float32)
    tmp134 = tmp112 * tmp133
    tmp135 = tmp134 * tmp52
    tmp136 = tmp130 + tmp135
    tmp137 = tmp112 * tmp112
    tmp138 = tmp115 * tmp115
    tmp139 = tmp138.to(tl.float32)
    tmp140 = tmp137 + tmp139
    tmp141 = tmp120 * tmp120
    tmp142 = tmp141.to(tl.float32)
    tmp143 = tmp137 + tmp142
    tmp144 = tmp140 + tmp143
    tmp145 = tmp126 * tmp126
    tmp146 = tmp145.to(tl.float32)
    tmp147 = tmp137 + tmp146
    tmp148 = tmp144 + tmp147
    tmp149 = tmp132 * tmp132
    tmp150 = tmp149.to(tl.float32)
    tmp151 = tmp137 + tmp150
    tmp152 = tmp148 + tmp151
    tmp154 = tmp6 == tmp5
    tmp155 = tmp154.to(tl.int64)
    tmp156 = tmp155.to(tl.float32)
    tmp157 = tmp153 * tmp156
    tmp158 = tmp157 * tmp14
    tmp159 = tmp19 == tmp5
    tmp160 = tmp159.to(tl.int64)
    tmp161 = tmp160.to(tl.float32)
    tmp162 = tmp153 * tmp161
    tmp163 = tmp162 * tmp26
    tmp164 = tmp158 + tmp163
    tmp165 = tmp32 == tmp5
    tmp166 = tmp165.to(tl.int64)
    tmp167 = tmp166.to(tl.float32)
    tmp168 = tmp153 * tmp167
    tmp169 = tmp168 * tmp39
    tmp170 = tmp164 + tmp169
    tmp171 = tmp45 == tmp5
    tmp172 = tmp171.to(tl.int64)
    tmp173 = tmp172.to(tl.float32)
    tmp174 = tmp153 * tmp173
    tmp175 = tmp174 * tmp52
    tmp176 = tmp170 + tmp175
    tmp177 = tmp153 * tmp153
    tmp178 = tmp155 * tmp155
    tmp179 = tmp178.to(tl.float32)
    tmp180 = tmp177 + tmp179
    tmp181 = tmp160 * tmp160
    tmp182 = tmp181.to(tl.float32)
    tmp183 = tmp177 + tmp182
    tmp184 = tmp180 + tmp183
    tmp185 = tmp166 * tmp166
    tmp186 = tmp185.to(tl.float32)
    tmp187 = tmp177 + tmp186
    tmp188 = tmp184 + tmp187
    tmp189 = tmp172 * tmp172
    tmp190 = tmp189.to(tl.float32)
    tmp191 = tmp177 + tmp190
    tmp192 = tmp188 + tmp191
    tl.store(out_ptr0 + (x0), tmp54, xmask)
    tl.store(out_ptr1 + (x0), tmp70, xmask)
    tl.store(out_ptr2 + (x0), tmp95, xmask)
    tl.store(out_ptr3 + (x0), tmp111, xmask)
    tl.store(out_ptr4 + (x0), tmp136, xmask)
    tl.store(out_ptr5 + (x0), tmp152, xmask)
    tl.store(out_ptr6 + (x0), tmp176, xmask)
    tl.store(out_ptr7 + (x0), tmp192, xmask)
