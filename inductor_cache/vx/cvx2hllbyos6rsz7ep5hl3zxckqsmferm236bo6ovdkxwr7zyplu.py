
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 14, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (tmp8 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.5
    tmp13 = tmp11 + tmp12
    tmp14 = 0.875
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 - tmp12
    tmp17 = libdevice.floor(tmp16)
    tmp18 = tmp16 - tmp17
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = 1.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp22 + tmp21
    tmp24 = -0.75
    tmp25 = tmp23 * tmp24
    tmp26 = -3.75
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp23
    tmp29 = -6.0
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30 * tmp23
    tmp32 = -3.0
    tmp33 = tmp31 - tmp32
    tmp34 = tmp9 * tmp33
    tmp36 = tmp35 + tmp1
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tmp39 = tl.load(in_ptr2 + (tmp38 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
    tmp40 = 1.25
    tmp41 = tmp22 * tmp40
    tmp42 = 2.25
    tmp43 = tmp41 - tmp42
    tmp44 = tmp43 * tmp22
    tmp45 = tmp44 * tmp22
    tmp46 = tmp45 + tmp21
    tmp47 = tmp39 * tmp46
    tmp48 = tmp34 + tmp47
    tmp50 = tmp49 + tmp1
    tmp51 = tmp49 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp49)
    tmp53 = tl.load(in_ptr2 + (tmp52 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
    tmp54 = tmp21 - tmp22
    tmp55 = tmp54 * tmp40
    tmp56 = tmp55 - tmp42
    tmp57 = tmp56 * tmp54
    tmp58 = tmp57 * tmp54
    tmp59 = tmp58 + tmp21
    tmp60 = tmp53 * tmp59
    tmp61 = tmp48 + tmp60
    tmp63 = tmp62 + tmp1
    tmp64 = tmp62 < 0
    tmp65 = tl.where(tmp64, tmp63, tmp62)
    tmp66 = tl.load(in_ptr2 + (tmp65 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
    tmp67 = 2.0
    tmp68 = tmp67 - tmp22
    tmp69 = tmp68 * tmp24
    tmp70 = tmp69 - tmp26
    tmp71 = tmp70 * tmp68
    tmp72 = tmp71 + tmp29
    tmp73 = tmp72 * tmp68
    tmp74 = tmp73 - tmp32
    tmp75 = tmp66 * tmp74
    tmp76 = tmp61 + tmp75
    tmp78 = tmp77 + tmp1
    tmp79 = tmp77 < 0
    tmp80 = tl.where(tmp79, tmp78, tmp77)
    tmp81 = tl.load(in_ptr2 + (tmp8 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp82 = tmp81 * tmp33
    tmp83 = tl.load(in_ptr2 + (tmp38 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp84 = tmp83 * tmp46
    tmp85 = tmp82 + tmp84
    tmp86 = tl.load(in_ptr2 + (tmp52 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp87 = tmp86 * tmp59
    tmp88 = tmp85 + tmp87
    tmp89 = tl.load(in_ptr2 + (tmp65 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp90 = tmp89 * tmp74
    tmp91 = tmp88 + tmp90
    tmp93 = tmp92 + tmp1
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tmp96 = tl.load(in_ptr2 + (tmp8 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp97 = tmp96 * tmp33
    tmp98 = tl.load(in_ptr2 + (tmp38 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp99 = tmp98 * tmp46
    tmp100 = tmp97 + tmp99
    tmp101 = tl.load(in_ptr2 + (tmp52 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp102 = tmp101 * tmp59
    tmp103 = tmp100 + tmp102
    tmp104 = tl.load(in_ptr2 + (tmp65 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp105 = tmp104 * tmp74
    tmp106 = tmp103 + tmp105
    tmp108 = tmp107 + tmp1
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tl.load(in_ptr2 + (tmp8 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
    tmp112 = tmp111 * tmp33
    tmp113 = tl.load(in_ptr2 + (tmp38 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
    tmp114 = tmp113 * tmp46
    tmp115 = tmp112 + tmp114
    tmp116 = tl.load(in_ptr2 + (tmp52 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
    tmp117 = tmp116 * tmp59
    tmp118 = tmp115 + tmp117
    tmp119 = tl.load(in_ptr2 + (tmp65 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
    tmp120 = tmp119 * tmp74
    tmp121 = tmp118 + tmp120
    tmp122 = x1
    tmp123 = tmp122.to(tl.float32)
    tmp124 = tmp123 + tmp12
    tmp125 = tmp124 * tmp14
    tmp126 = tmp125 - tmp12
    tmp127 = libdevice.floor(tmp126)
    tmp128 = tmp126 - tmp127
    tmp129 = triton_helpers.maximum(tmp128, tmp19)
    tmp130 = triton_helpers.minimum(tmp129, tmp21)
    tmp131 = tmp130 + tmp21
    tmp132 = tmp131 * tmp24
    tmp133 = tmp132 - tmp26
    tmp134 = tmp133 * tmp131
    tmp135 = tmp134 + tmp29
    tmp136 = tmp135 * tmp131
    tmp137 = tmp136 - tmp32
    tmp138 = tmp76 * tmp137
    tmp139 = tmp130 * tmp40
    tmp140 = tmp139 - tmp42
    tmp141 = tmp140 * tmp130
    tmp142 = tmp141 * tmp130
    tmp143 = tmp142 + tmp21
    tmp144 = tmp91 * tmp143
    tmp145 = tmp138 + tmp144
    tmp146 = tmp21 - tmp130
    tmp147 = tmp146 * tmp40
    tmp148 = tmp147 - tmp42
    tmp149 = tmp148 * tmp146
    tmp150 = tmp149 * tmp146
    tmp151 = tmp150 + tmp21
    tmp152 = tmp106 * tmp151
    tmp153 = tmp145 + tmp152
    tmp154 = tmp67 - tmp130
    tmp155 = tmp154 * tmp24
    tmp156 = tmp155 - tmp26
    tmp157 = tmp156 * tmp154
    tmp158 = tmp157 + tmp29
    tmp159 = tmp158 * tmp154
    tmp160 = tmp159 - tmp32
    tmp161 = tmp121 * tmp160
    tmp162 = tmp153 + tmp161
    tl.store(in_out_ptr0 + (x4), tmp162, None)
