
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'out_ptr9': '*fp32', 'out_ptr10': '*fp32', 'out_ptr11': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_gather_mul_rsub_sub_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_gather_mul_rsub_sub_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    y3 = yindex // 4096
    y2 = (yindex % 4096)
    tmp0 = tl.load(in_ptr0 + (x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr0 + (9 + x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr1 + (9 + x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr2 + (9 + x1 + 18*y0), xmask, eviction_policy='evict_last')
    tmp170 = tl.load(in_ptr4 + (y2 + 4096*x1 + 36864*y3), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp5 = 1.0
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 - tmp3
    tmp11 = tmp5 - tmp10
    tmp12 = x1
    tmp13 = tl.full([1, 1], 0, tl.int64)
    tmp14 = tmp12 >= tmp13
    tmp15 = tl.full([1, 1], 9, tl.int64)
    tmp16 = tmp12 < tmp15
    tmp17 = tl.load(in_ptr0 + (18*y0 + (x1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17.to(tl.int64)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp16, tmp18, tmp19)
    tmp21 = tmp12 >= tmp15
    tmp22 = tl.full([1, 1], 18, tl.int64)
    tmp23 = tmp12 < tmp22
    tmp24 = tl.load(in_ptr2 + (9 + 18*y0 + ((-9) + x1)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24.to(tl.int64)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp21, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 - tmp3
    tmp31 = tmp30 + tmp5
    tmp32 = tl.load(in_ptr2 + (18*y0 + (x1)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp32.to(tl.int64)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp16, tmp33, tmp34)
    tmp36 = tl.load(in_ptr0 + (9 + 18*y0 + ((-9) + x1)), tmp21 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp36.to(tl.int64)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp21, tmp37, tmp38)
    tmp40 = tl.where(tmp16, tmp35, tmp39)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 - tmp3
    tmp43 = tmp5 - tmp42
    tmp45 = tmp44.to(tl.int64)
    tmp46 = tmp45.to(tl.float32)
    tmp48 = tmp46 - tmp47
    tmp49 = tmp48 + tmp5
    tmp51 = tmp50.to(tl.int64)
    tmp52 = tmp51.to(tl.float32)
    tmp53 = tmp52 - tmp47
    tmp54 = tmp5 - tmp53
    tmp55 = 9 + x1
    tmp56 = tmp55 >= tmp13
    tmp57 = tmp55 < tmp15
    tmp58 = tl.load(in_ptr0 + (18*y0 + (9 + x1)), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp58.to(tl.int64)
    tmp60 = tl.full(tmp59.shape, 0.0, tmp59.dtype)
    tmp61 = tl.where(tmp57, tmp59, tmp60)
    tmp62 = tmp55 >= tmp15
    tmp63 = tmp55 < tmp22
    tmp64 = tl.load(in_ptr2 + (9 + 18*y0 + (x1)), tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp64.to(tl.int64)
    tmp66 = tl.full(tmp65.shape, 0.0, tmp65.dtype)
    tmp67 = tl.where(tmp62, tmp65, tmp66)
    tmp68 = tl.where(tmp57, tmp61, tmp67)
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp69 - tmp47
    tmp71 = tmp5 - tmp70
    tmp72 = tl.load(in_ptr2 + (18*y0 + (9 + x1)), tmp57 & xmask, eviction_policy='evict_last', other=0.0)
    tmp73 = tmp72.to(tl.int64)
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp57, tmp73, tmp74)
    tmp76 = tl.load(in_ptr0 + (9 + 18*y0 + (x1)), tmp62 & xmask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp76.to(tl.int64)
    tmp78 = tl.full(tmp77.shape, 0.0, tmp77.dtype)
    tmp79 = tl.where(tmp62, tmp77, tmp78)
    tmp80 = tl.where(tmp57, tmp75, tmp79)
    tmp81 = tmp80.to(tl.float32)
    tmp82 = tmp81 - tmp47
    tmp83 = tmp82 + tmp5
    tmp84 = tl.full([1, 1], 66, tl.int64)
    tmp85 = tmp1 * tmp84
    tmp86 = tmp85 + tmp45
    tmp87 = tl.full([XBLOCK, YBLOCK], 4356, tl.int32)
    tmp88 = tmp86 + tmp87
    tmp89 = tmp86 < 0
    tmp90 = tl.where(tmp89, tmp88, tmp86)
    tl.device_assert(((0 <= tmp90) & (tmp90 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp90 < 4356")
    tmp92 = (-1) + (((tmp90 // 66) % 66))
    tmp93 = tmp92.to(tl.int32)
    tmp94 = tmp93 >= tmp13
    tmp95 = tl.full([1, 1], 64, tl.int64)
    tmp96 = tmp93 < tmp95
    tmp97 = (-1) + ((tmp90 % 66))
    tmp98 = tmp97.to(tl.int32)
    tmp99 = tmp98 >= tmp13
    tmp100 = tmp98 < tmp95
    tmp101 = tmp94 & tmp96
    tmp102 = tmp101 & tmp99
    tmp103 = tmp102 & tmp100
    tmp104 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp90 // 66) % 66)) + 4096*y3 + ((tmp90 % 66)), [XBLOCK, YBLOCK])), tmp103 & xmask, eviction_policy='evict_last', other=0.0)
    tmp105 = tmp8 * tmp84
    tmp106 = tmp105 + tmp51
    tmp107 = tmp106 + tmp87
    tmp108 = tmp106 < 0
    tmp109 = tl.where(tmp108, tmp107, tmp106)
    tl.device_assert(((0 <= tmp109) & (tmp109 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp109 < 4356")
    tmp111 = (-1) + (((tmp109 // 66) % 66))
    tmp112 = tmp111.to(tl.int32)
    tmp113 = tmp112 >= tmp13
    tmp114 = tmp112 < tmp95
    tmp115 = (-1) + ((tmp109 % 66))
    tmp116 = tmp115.to(tl.int32)
    tmp117 = tmp116 >= tmp13
    tmp118 = tmp116 < tmp95
    tmp119 = tmp113 & tmp114
    tmp120 = tmp119 & tmp117
    tmp121 = tmp120 & tmp118
    tmp122 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp109 // 66) % 66)) + 4096*y3 + ((tmp109 % 66)), [XBLOCK, YBLOCK])), tmp121 & xmask, eviction_policy='evict_last', other=0.0)
    tmp123 = tmp40 * tmp84
    tmp124 = tmp123 + tmp80
    tmp125 = tmp124 + tmp87
    tmp126 = tmp124 < 0
    tmp127 = tl.where(tmp126, tmp125, tmp124)
    tl.device_assert(((0 <= tmp127) & (tmp127 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp127 < 4356")
    tmp129 = (-1) + (((tmp127 // 66) % 66))
    tmp130 = tmp129.to(tl.int32)
    tmp131 = tmp130 >= tmp13
    tmp132 = tmp130 < tmp95
    tmp133 = (-1) + ((tmp127 % 66))
    tmp134 = tmp133.to(tl.int32)
    tmp135 = tmp134 >= tmp13
    tmp136 = tmp134 < tmp95
    tmp137 = tmp131 & tmp132
    tmp138 = tmp137 & tmp135
    tmp139 = tmp138 & tmp136
    tmp140 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp127 // 66) % 66)) + 4096*y3 + ((tmp127 % 66)), [XBLOCK, YBLOCK])), tmp139 & xmask, eviction_policy='evict_last', other=0.0)
    tmp141 = tmp28 * tmp84
    tmp142 = tmp141 + tmp68
    tmp143 = tmp142 + tmp87
    tmp144 = tmp142 < 0
    tmp145 = tl.where(tmp144, tmp143, tmp142)
    tl.device_assert(((0 <= tmp145) & (tmp145 < 4356)) | ~(xmask), "index out of bounds: 0 <= tmp145 < 4356")
    tmp147 = (-1) + (((tmp145 // 66) % 66))
    tmp148 = tmp147.to(tl.int32)
    tmp149 = tmp148 >= tmp13
    tmp150 = tmp148 < tmp95
    tmp151 = (-1) + ((tmp145 % 66))
    tmp152 = tmp151.to(tl.int32)
    tmp153 = tmp152 >= tmp13
    tmp154 = tmp152 < tmp95
    tmp155 = tmp149 & tmp150
    tmp156 = tmp155 & tmp153
    tmp157 = tmp156 & tmp154
    tmp158 = tl.load(in_ptr3 + (tl.broadcast_to((-65) + 64*(((tmp145 // 66) % 66)) + 4096*y3 + ((tmp145 % 66)), [XBLOCK, YBLOCK])), tmp157 & xmask, eviction_policy='evict_last', other=0.0)
    tmp159 = tmp6 * tmp49
    tmp160 = tmp159 * tmp104
    tmp161 = tmp11 * tmp54
    tmp162 = tmp161 * tmp122
    tmp163 = tmp160 + tmp162
    tmp164 = tmp31 * tmp71
    tmp165 = tmp164 * tmp158
    tmp166 = tmp163 + tmp165
    tmp167 = tmp43 * tmp83
    tmp168 = tmp167 * tmp140
    tmp169 = tmp166 + tmp168
    tmp171 = tl.sigmoid(tmp170)
    tmp172 = tmp169 * tmp171
    tl.store(out_ptr0 + (x1 + 9*y0), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + 9*y0), tmp11, xmask)
    tl.store(out_ptr2 + (x1 + 9*y0), tmp31, xmask)
    tl.store(out_ptr3 + (x1 + 9*y0), tmp43, xmask)
    tl.store(out_ptr4 + (x1 + 9*y0), tmp49, xmask)
    tl.store(out_ptr5 + (x1 + 9*y0), tmp54, xmask)
    tl.store(out_ptr6 + (x1 + 9*y0), tmp71, xmask)
    tl.store(out_ptr7 + (x1 + 9*y0), tmp83, xmask)
    tl.store(out_ptr8 + (x1 + 9*y0), tmp104, xmask)
    tl.store(out_ptr9 + (x1 + 9*y0), tmp122, xmask)
    tl.store(out_ptr10 + (x1 + 9*y0), tmp140, xmask)
    tl.store(out_ptr11 + (x1 + 9*y0), tmp158, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 9*y0), tmp172, xmask)
