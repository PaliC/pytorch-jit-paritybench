
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*i64', 'out_ptr2': '*fp32', 'out_ptr3': '*i64', 'out_ptr4': '*i64', 'out_ptr5': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_grid_sampler_2d_2(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 16)
    x2 = xindex // 64
    x3 = xindex
    x4 = xindex // 16
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 32*x2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (16 + x0 + 32*x2), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr1 + (x0 + 32*x2), None, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int32)
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.3333333333333333
    tmp7 = tmp5 * tmp6
    tmp8 = 1.0
    tmp9 = tmp7 - tmp8
    tmp11 = tl.where(tmp2, tmp9, tmp10)
    tmp12 = tmp11 * tmp4
    tmp13 = 1.5
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.floor(tmp14)
    tmp16 = 0.0
    tmp17 = tmp15 >= tmp16
    tmp18 = 4.0
    tmp19 = tmp15 < tmp18
    tmp20 = tmp1 == tmp1
    tmp21 = tl.where(tmp20, tmp9, tmp3)
    tmp22 = tmp21 * tmp4
    tmp23 = tmp22 + tmp13
    tmp24 = libdevice.floor(tmp23)
    tmp25 = tmp24 >= tmp16
    tmp26 = tmp24 < tmp18
    tmp27 = tmp25 & tmp26
    tmp28 = tmp19 & tmp27
    tmp29 = tmp17 & tmp28
    tmp30 = tmp24.to(tl.int64)
    tmp31 = tl.full([1], 0, tl.int64)
    tmp32 = tl.where(tmp29, tmp30, tmp31)
    tmp33 = tmp15.to(tl.int64)
    tmp34 = tl.where(tmp29, tmp33, tmp31)
    tmp35 = tmp15 + tmp8
    tmp36 = tmp35 - tmp14
    tmp37 = tmp24 + tmp8
    tmp38 = tmp37 - tmp23
    tmp39 = tmp36 * tmp38
    tmp40 = tl.where(tmp29, tmp39, tmp16)
    tmp42 = tmp41 * tmp4
    tmp43 = tmp42 * tmp6
    tmp44 = tmp43 - tmp8
    tmp46 = tl.where(tmp2, tmp44, tmp45)
    tmp47 = tmp46 * tmp4
    tmp48 = tmp47 + tmp13
    tmp49 = libdevice.floor(tmp48)
    tmp50 = tmp49 >= tmp16
    tmp51 = tmp49 < tmp18
    tmp52 = tl.where(tmp20, tmp44, tmp41)
    tmp53 = tmp52 * tmp4
    tmp54 = tmp53 + tmp13
    tmp55 = libdevice.floor(tmp54)
    tmp56 = tmp55 >= tmp16
    tmp57 = tmp55 < tmp18
    tmp58 = tmp56 & tmp57
    tmp59 = tmp51 & tmp58
    tmp60 = tmp50 & tmp59
    tmp61 = tmp55.to(tl.int64)
    tmp62 = tl.where(tmp60, tmp61, tmp31)
    tmp63 = tmp49.to(tl.int64)
    tmp64 = tl.where(tmp60, tmp63, tmp31)
    tmp65 = tmp49 + tmp8
    tmp66 = tmp65 - tmp48
    tmp67 = tmp55 + tmp8
    tmp68 = tmp67 - tmp54
    tmp69 = tmp66 * tmp68
    tmp70 = tl.where(tmp60, tmp69, tmp16)
    tmp71 = tmp35 < tmp18
    tmp72 = tmp37 >= tmp16
    tmp73 = tmp37 < tmp18
    tmp74 = tmp72 & tmp73
    tmp75 = tmp71 & tmp74
    tmp76 = tmp19 & tmp74
    tmp77 = tmp17 & tmp76
    tmp78 = tmp35 >= tmp16
    tmp79 = tmp71 & tmp27
    tmp80 = tmp78 & tmp79
    tmp81 = tmp65 < tmp18
    tmp82 = tmp67 >= tmp16
    tmp83 = tmp67 < tmp18
    tmp84 = tmp82 & tmp83
    tmp85 = tmp81 & tmp84
    tmp86 = tmp51 & tmp84
    tmp87 = tmp50 & tmp86
    tmp88 = tmp65 >= tmp16
    tmp89 = tmp81 & tmp58
    tmp90 = tmp88 & tmp89
    tmp91 = tl.where(tmp90, tmp61, tmp31)
    tmp92 = tl.full([XBLOCK], 4, tl.int32)
    tmp93 = tmp91 + tmp92
    tmp94 = tmp91 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp91)
    tl.device_assert((0 <= tmp95) & (tmp95 < 4), "index out of bounds: 0 <= tmp95 < 4")
    tmp97 = tmp65.to(tl.int64)
    tmp98 = tl.where(tmp90, tmp97, tmp31)
    tmp99 = tmp98 + tmp92
    tmp100 = tmp98 < 0
    tmp101 = tl.where(tmp100, tmp99, tmp98)
    tl.device_assert((0 <= tmp101) & (tmp101 < 4), "index out of bounds: 0 <= tmp101 < 4")
    tmp103 = tl.load(in_ptr2 + (tmp101 + 4*tmp95 + 16*x4), None, eviction_policy='evict_last')
    tmp104 = tmp48 - tmp49
    tmp105 = tmp104 * tmp68
    tmp106 = tl.where(tmp90, tmp105, tmp16)
    tmp107 = tmp103 * tmp106
    tmp108 = tmp67.to(tl.int64)
    tmp109 = tl.where(tmp87, tmp108, tmp31)
    tmp110 = tmp109 + tmp92
    tmp111 = tmp109 < 0
    tmp112 = tl.where(tmp111, tmp110, tmp109)
    tl.device_assert((0 <= tmp112) & (tmp112 < 4), "index out of bounds: 0 <= tmp112 < 4")
    tmp114 = tl.where(tmp87, tmp63, tmp31)
    tmp115 = tmp114 + tmp92
    tmp116 = tmp114 < 0
    tmp117 = tl.where(tmp116, tmp115, tmp114)
    tl.device_assert((0 <= tmp117) & (tmp117 < 4), "index out of bounds: 0 <= tmp117 < 4")
    tmp119 = tl.load(in_ptr2 + (tmp117 + 4*tmp112 + 16*x4), None, eviction_policy='evict_last')
    tmp120 = tmp54 - tmp55
    tmp121 = tmp66 * tmp120
    tmp122 = tl.where(tmp87, tmp121, tmp16)
    tmp123 = tmp119 * tmp122
    tmp124 = tmp88 & tmp85
    tmp125 = tmp104 * tmp120
    tmp126 = tl.where(tmp124, tmp125, tmp16)
    tmp127 = tl.where(tmp124, tmp108, tmp31)
    tmp128 = tmp127 + tmp92
    tmp129 = tmp127 < 0
    tmp130 = tl.where(tmp129, tmp128, tmp127)
    tl.device_assert((0 <= tmp130) & (tmp130 < 4), "index out of bounds: 0 <= tmp130 < 4")
    tmp132 = tl.where(tmp124, tmp97, tmp31)
    tmp133 = tmp132 + tmp92
    tmp134 = tmp132 < 0
    tmp135 = tl.where(tmp134, tmp133, tmp132)
    tl.device_assert((0 <= tmp135) & (tmp135 < 4), "index out of bounds: 0 <= tmp135 < 4")
    tmp137 = tl.load(in_ptr2 + (tmp135 + 4*tmp130 + 16*x4), None, eviction_policy='evict_last')
    tmp138 = tmp137 * tmp126
    tmp139 = tl.where(tmp80, tmp30, tmp31)
    tmp140 = tmp139 + tmp92
    tmp141 = tmp139 < 0
    tmp142 = tl.where(tmp141, tmp140, tmp139)
    tl.device_assert((0 <= tmp142) & (tmp142 < 4), "index out of bounds: 0 <= tmp142 < 4")
    tmp144 = tmp35.to(tl.int64)
    tmp145 = tl.where(tmp80, tmp144, tmp31)
    tmp146 = tmp145 + tmp92
    tmp147 = tmp145 < 0
    tmp148 = tl.where(tmp147, tmp146, tmp145)
    tl.device_assert((0 <= tmp148) & (tmp148 < 4), "index out of bounds: 0 <= tmp148 < 4")
    tmp150 = tl.load(in_ptr2 + (tmp148 + 4*tmp142 + 16*x4), None, eviction_policy='evict_last')
    tmp151 = tmp14 - tmp15
    tmp152 = tmp151 * tmp38
    tmp153 = tl.where(tmp80, tmp152, tmp16)
    tmp154 = tmp150 * tmp153
    tmp155 = tmp37.to(tl.int64)
    tmp156 = tl.where(tmp77, tmp155, tmp31)
    tmp157 = tmp156 + tmp92
    tmp158 = tmp156 < 0
    tmp159 = tl.where(tmp158, tmp157, tmp156)
    tl.device_assert((0 <= tmp159) & (tmp159 < 4), "index out of bounds: 0 <= tmp159 < 4")
    tmp161 = tl.where(tmp77, tmp33, tmp31)
    tmp162 = tmp161 + tmp92
    tmp163 = tmp161 < 0
    tmp164 = tl.where(tmp163, tmp162, tmp161)
    tl.device_assert((0 <= tmp164) & (tmp164 < 4), "index out of bounds: 0 <= tmp164 < 4")
    tmp166 = tl.load(in_ptr2 + (tmp164 + 4*tmp159 + 16*x4), None, eviction_policy='evict_last')
    tmp167 = tmp23 - tmp24
    tmp168 = tmp36 * tmp167
    tmp169 = tl.where(tmp77, tmp168, tmp16)
    tmp170 = tmp166 * tmp169
    tmp171 = tmp78 & tmp75
    tmp172 = tmp151 * tmp167
    tmp173 = tl.where(tmp171, tmp172, tmp16)
    tmp174 = tl.where(tmp171, tmp155, tmp31)
    tmp175 = tmp174 + tmp92
    tmp176 = tmp174 < 0
    tmp177 = tl.where(tmp176, tmp175, tmp174)
    tl.device_assert((0 <= tmp177) & (tmp177 < 4), "index out of bounds: 0 <= tmp177 < 4")
    tmp179 = tl.where(tmp171, tmp144, tmp31)
    tmp180 = tmp179 + tmp92
    tmp181 = tmp179 < 0
    tmp182 = tl.where(tmp181, tmp180, tmp179)
    tl.device_assert((0 <= tmp182) & (tmp182 < 4), "index out of bounds: 0 <= tmp182 < 4")
    tmp184 = tl.load(in_ptr2 + (tmp182 + 4*tmp177 + 16*x4), None, eviction_policy='evict_last')
    tmp185 = tmp184 * tmp173
    tl.store(out_ptr0 + (x3), tmp32, None)
    tl.store(out_ptr1 + (x3), tmp34, None)
    tl.store(out_ptr2 + (x3), tmp40, None)
    tl.store(out_ptr3 + (x3), tmp62, None)
    tl.store(out_ptr4 + (x3), tmp64, None)
    tl.store(out_ptr5 + (x3), tmp70, None)
    tl.store(in_out_ptr0 + (x3), tmp107, None)
    tl.store(in_out_ptr1 + (x3), tmp123, None)
    tl.store(in_out_ptr2 + (x3), tmp138, None)
    tl.store(in_out_ptr3 + (x3), tmp154, None)
    tl.store(in_out_ptr4 + (x3), tmp170, None)
    tl.store(in_out_ptr5 + (x3), tmp185, None)
