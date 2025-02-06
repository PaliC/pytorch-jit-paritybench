
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*i8', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_max_pool2d_with_indices_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 38, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_avg_pool2d_max_pool2d_with_indices_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr3, out_ptr4, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 2) % 2)
    x0 = (xindex % 2)
    x8 = xindex // 2
    x6 = xindex
    x4 = ((xindex // 4) % 42)
    x5 = xindex // 168
    x7 = (xindex % 168)
    tmp96 = tl.load(in_ptr1 + (x6), xmask)
    tmp97 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp99 = tl.load(in_ptr3 + (x4), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr4 + (x4), xmask, eviction_policy='evict_last')
    tmp110 = tl.load(in_ptr5 + (x4), xmask, eviction_policy='evict_last')
    tmp113 = tl.load(in_ptr6 + (x6), xmask)
    tmp114 = tl.load(in_ptr7 + (x4), xmask, eviction_policy='evict_last')
    tmp116 = tl.load(in_ptr8 + (x4), xmask, eviction_policy='evict_last')
    tmp122 = tl.load(in_ptr9 + (x4), xmask, eviction_policy='evict_last')
    tmp124 = tl.load(in_ptr10 + (x4), xmask, eviction_policy='evict_last')
    tmp180 = tl.load(in_ptr11 + (x6), xmask)
    tmp0 = (-1) + 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x8), tmp10 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = 2*x0
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x8), tmp16 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x0
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3) + 2*x0 + 8*x8), tmp23 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x8), tmp30 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (2*x0 + 8*x8), tmp33 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x8), tmp36 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x1
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3 + 2*x0 + 8*x8), tmp43 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x8), tmp46 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x8), tmp49 & xmask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tl.load(in_ptr0 + ((-5) + 2*x0 + 8*x8), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.load(in_ptr0 + ((-4) + 2*x0 + 8*x8), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp78 + tmp77
    tmp80 = tl.load(in_ptr0 + ((-3) + 2*x0 + 8*x8), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp79
    tmp82 = tl.load(in_ptr0 + ((-1) + 2*x0 + 8*x8), tmp30 & xmask, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp82 + tmp81
    tmp84 = tl.load(in_ptr0 + (2*x0 + 8*x8), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp84 + tmp83
    tmp86 = tl.load(in_ptr0 + (1 + 2*x0 + 8*x8), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp86 + tmp85
    tmp88 = tl.load(in_ptr0 + (3 + 2*x0 + 8*x8), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp88 + tmp87
    tmp90 = tl.load(in_ptr0 + (4 + 2*x0 + 8*x8), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp90 + tmp89
    tmp92 = tl.load(in_ptr0 + (5 + 2*x0 + 8*x8), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp93 = tmp92 + tmp91
    tmp94 = ((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0))) + ((4) * ((4) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (4)))*((4) * ((4) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (4))) + ((-1)*((0) * ((0) >= ((-1) + 2*x0)) + ((-1) + 2*x0) * (((-1) + 2*x0) > (0)))*((4) * ((4) <= (2 + 2*x1)) + (2 + 2*x1) * ((2 + 2*x1) < (4)))) + ((-1)*((0) * ((0) >= ((-1) + 2*x1)) + ((-1) + 2*x1) * (((-1) + 2*x1) > (0)))*((4) * ((4) <= (2 + 2*x0)) + (2 + 2*x0) * ((2 + 2*x0) < (4))))
    tmp95 = tmp93 / tmp94
    tmp98 = tmp96 - tmp97
    tmp100 = 0.001
    tmp101 = tmp99 + tmp100
    tmp102 = libdevice.sqrt(tmp101)
    tmp103 = tl.full([1], 1, tl.int32)
    tmp104 = tmp103 / tmp102
    tmp105 = 1.0
    tmp106 = tmp104 * tmp105
    tmp107 = tmp98 * tmp106
    tmp109 = tmp107 * tmp108
    tmp111 = tmp109 + tmp110
    tmp112 = tmp51 + tmp111
    tmp115 = tmp113 - tmp114
    tmp117 = tmp116 + tmp100
    tmp118 = libdevice.sqrt(tmp117)
    tmp119 = tmp103 / tmp118
    tmp120 = tmp119 * tmp105
    tmp121 = tmp115 * tmp120
    tmp123 = tmp121 * tmp122
    tmp125 = tmp123 + tmp124
    tmp126 = tmp95 + tmp125
    tmp127 = (-1) + x1
    tmp128 = tmp127 >= tmp1
    tmp129 = tl.full([1], 2, tl.int64)
    tmp130 = tmp127 < tmp129
    tmp131 = tmp128 & tmp130
    tmp132 = (-1) + x0
    tmp133 = tmp132 >= tmp1
    tmp134 = tmp132 < tmp129
    tmp135 = tmp133 & tmp134
    tmp136 = tmp131 & tmp135
    tmp137 = tl.load(in_ptr11 + ((-3) + x6), tmp136 & xmask, other=0.0)
    tmp138 = x0
    tmp139 = tmp138 >= tmp1
    tmp140 = tmp138 < tmp129
    tmp141 = tmp139 & tmp140
    tmp142 = tmp131 & tmp141
    tmp143 = tl.load(in_ptr11 + ((-2) + x6), tmp142 & xmask, other=0.0)
    tmp144 = tmp143 + tmp137
    tmp145 = 1 + x0
    tmp146 = tmp145 >= tmp1
    tmp147 = tmp145 < tmp129
    tmp148 = tmp146 & tmp147
    tmp149 = tmp131 & tmp148
    tmp150 = tl.load(in_ptr11 + ((-1) + x6), tmp149 & xmask, other=0.0)
    tmp151 = tmp150 + tmp144
    tmp152 = x1
    tmp153 = tmp152 >= tmp1
    tmp154 = tmp152 < tmp129
    tmp155 = tmp153 & tmp154
    tmp156 = tmp155 & tmp135
    tmp157 = tl.load(in_ptr11 + ((-1) + x6), tmp156 & xmask, other=0.0)
    tmp158 = tmp157 + tmp151
    tmp159 = tmp155 & tmp141
    tmp160 = tl.load(in_ptr11 + (x6), tmp159 & xmask, other=0.0)
    tmp161 = tmp160 + tmp158
    tmp162 = tmp155 & tmp148
    tmp163 = tl.load(in_ptr11 + (1 + x6), tmp162 & xmask, other=0.0)
    tmp164 = tmp163 + tmp161
    tmp165 = 1 + x1
    tmp166 = tmp165 >= tmp1
    tmp167 = tmp165 < tmp129
    tmp168 = tmp166 & tmp167
    tmp169 = tmp168 & tmp135
    tmp170 = tl.load(in_ptr11 + (1 + x6), tmp169 & xmask, other=0.0)
    tmp171 = tmp170 + tmp164
    tmp172 = tmp168 & tmp141
    tmp173 = tl.load(in_ptr11 + (2 + x6), tmp172 & xmask, other=0.0)
    tmp174 = tmp173 + tmp171
    tmp175 = tmp168 & tmp148
    tmp176 = tl.load(in_ptr11 + (3 + x6), tmp175 & xmask, other=0.0)
    tmp177 = tmp176 + tmp174
    tmp178 = 4 + ((-2)*((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))) + ((-2)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))) + ((0) * ((0) >= ((-1) + x0)) + ((-1) + x0) * (((-1) + x0) > (0)))*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))
    tmp179 = tmp177 / tmp178
    tmp181 = tl.full([1], 0, tl.int32)
    tmp182 = triton_helpers.maximum(tmp181, tmp180)
    tmp183 = tmp179 + tmp112
    tl.store(out_ptr0 + (x6), tmp51, xmask)
    tl.store(out_ptr1 + (x6), tmp76, xmask)
    tl.store(out_ptr3 + (x7 + 672*x5), tmp112, xmask)
    tl.store(out_ptr4 + (x7 + 672*x5), tmp126, xmask)
    tl.store(out_ptr6 + (x6), tmp182, xmask)
    tl.store(out_ptr7 + (x7 + 672*x5), tmp183, xmask)
