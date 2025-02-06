
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*i64', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*i64', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*i64', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*i64', 'in_ptr25': '*i64', 'in_ptr26': '*fp32', 'in_ptr27': '*i64', 'in_ptr28': '*fp32', 'in_ptr29': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 27, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5242880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 4096) % 320)
    x3 = xindex // 1310720
    x4 = (xindex % 4096)
    x1 = ((xindex // 64) % 64)
    x0 = (xindex % 64)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 4096*(x2) + 262144*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 192, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 4096*((-64) + x2) + 524288*x3), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 224, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x4 + 4096*((-192) + x2) + 131072*x3), tmp14, other=0.0)
    tmp16 = tl.load(in_ptr3 + (x1), tmp14, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.full([XBLOCK], 8, tl.int32)
    tmp18 = tmp16 + tmp17
    tmp19 = tmp16 < 0
    tmp20 = tl.where(tmp19, tmp18, tmp16)
    tmp21 = tl.load(in_ptr4 + (x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp17
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr5 + (tmp24 + 8*tmp20 + 64*((-192) + x2) + 2048*x3), tmp14, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr6 + (x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp26 + tmp17
    tmp28 = tmp26 < 0
    tmp29 = tl.where(tmp28, tmp27, tmp26)
    tmp30 = tl.load(in_ptr5 + (tmp29 + 8*tmp20 + 64*((-192) + x2) + 2048*x3), tmp14, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 - tmp25
    tmp32 = tl.load(in_ptr7 + (x0), tmp14, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp25 + tmp33
    tmp35 = tmp34 - tmp15
    tmp36 = tl.load(in_ptr8 + (x1), tmp14, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp15 + tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp14, tmp38, tmp39)
    tmp41 = tmp0 >= tmp12
    tmp42 = tl.full([1], 256, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tmp41 & tmp43
    tmp45 = tl.load(in_ptr9 + (x4 + 4096*((-224) + x2) + 131072*x3), tmp44, other=0.0)
    tmp46 = tl.load(in_ptr10 + (x1), tmp44, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.full([XBLOCK], 4, tl.int32)
    tmp48 = tmp46 + tmp47
    tmp49 = tmp46 < 0
    tmp50 = tl.where(tmp49, tmp48, tmp46)
    tmp51 = tl.load(in_ptr11 + (x0), tmp44, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp47
    tmp53 = tmp51 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp51)
    tmp55 = tl.load(in_ptr12 + (tmp54 + 4*tmp50 + 16*((-224) + x2) + 512*x3), tmp44, eviction_policy='evict_last', other=0.0)
    tmp56 = tl.load(in_ptr13 + (x0), tmp44, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp56 + tmp47
    tmp58 = tmp56 < 0
    tmp59 = tl.where(tmp58, tmp57, tmp56)
    tmp60 = tl.load(in_ptr12 + (tmp59 + 4*tmp50 + 16*((-224) + x2) + 512*x3), tmp44, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp60 - tmp55
    tmp62 = tl.load(in_ptr14 + (x0), tmp44, eviction_policy='evict_last', other=0.0)
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 + tmp63
    tmp65 = tmp64 - tmp45
    tmp66 = tl.load(in_ptr15 + (x1), tmp44, eviction_policy='evict_last', other=0.0)
    tmp67 = tmp65 * tmp66
    tmp68 = tmp45 + tmp67
    tmp69 = tl.full(tmp68.shape, 0.0, tmp68.dtype)
    tmp70 = tl.where(tmp44, tmp68, tmp69)
    tmp71 = tmp0 >= tmp42
    tmp72 = tl.full([1], 288, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tl.load(in_ptr16 + (x4 + 4096*((-256) + x2) + 131072*x3), tmp74, other=0.0)
    tmp76 = tl.load(in_ptr17 + (x1), tmp74, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.full([XBLOCK], 2, tl.int32)
    tmp78 = tmp76 + tmp77
    tmp79 = tmp76 < 0
    tmp80 = tl.where(tmp79, tmp78, tmp76)
    tmp81 = tl.load(in_ptr18 + (x0), tmp74, eviction_policy='evict_last', other=0.0)
    tmp82 = tmp81 + tmp77
    tmp83 = tmp81 < 0
    tmp84 = tl.where(tmp83, tmp82, tmp81)
    tmp85 = tl.load(in_ptr19 + (tmp84 + 2*tmp80 + 4*((-256) + x2) + 128*x3), tmp74, eviction_policy='evict_last', other=0.0)
    tmp86 = tl.load(in_ptr20 + (x0), tmp74, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp86 + tmp77
    tmp88 = tmp86 < 0
    tmp89 = tl.where(tmp88, tmp87, tmp86)
    tmp90 = tl.load(in_ptr19 + (tmp89 + 2*tmp80 + 4*((-256) + x2) + 128*x3), tmp74, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp90 - tmp85
    tmp92 = tl.load(in_ptr21 + (x0), tmp74, eviction_policy='evict_last', other=0.0)
    tmp93 = tmp91 * tmp92
    tmp94 = tmp85 + tmp93
    tmp95 = tmp94 - tmp75
    tmp96 = tl.load(in_ptr22 + (x1), tmp74, eviction_policy='evict_last', other=0.0)
    tmp97 = tmp95 * tmp96
    tmp98 = tmp75 + tmp97
    tmp99 = tl.full(tmp98.shape, 0.0, tmp98.dtype)
    tmp100 = tl.where(tmp74, tmp98, tmp99)
    tmp101 = tmp0 >= tmp72
    tmp102 = tl.full([1], 320, tl.int64)
    tmp103 = tmp0 < tmp102
    tmp104 = tl.load(in_ptr23 + (x4 + 4096*((-288) + x2) + 131072*x3), tmp101, other=0.0)
    tmp105 = tl.load(in_ptr24 + (x1), tmp101, eviction_policy='evict_last', other=0.0)
    tmp106 = tl.full([XBLOCK], 1, tl.int32)
    tmp107 = tmp105 + tmp106
    tmp108 = tmp105 < 0
    tmp109 = tl.where(tmp108, tmp107, tmp105)
    tmp110 = tl.load(in_ptr25 + (x0), tmp101, eviction_policy='evict_last', other=0.0)
    tmp111 = tmp110 + tmp106
    tmp112 = tmp110 < 0
    tmp113 = tl.where(tmp112, tmp111, tmp110)
    tmp114 = tl.load(in_ptr26 + (32*x3 + ((-288) + x2)), tmp101, eviction_policy='evict_last', other=0.0)
    tmp115 = tl.load(in_ptr27 + (x0), tmp101, eviction_policy='evict_last', other=0.0)
    tmp116 = tmp115 + tmp106
    tmp117 = tmp115 < 0
    tmp118 = tl.where(tmp117, tmp116, tmp115)
    tmp119 = tmp114 - tmp114
    tmp120 = tl.load(in_ptr28 + (x0), tmp101, eviction_policy='evict_last', other=0.0)
    tmp121 = tmp119 * tmp120
    tmp122 = tmp114 + tmp121
    tmp123 = tmp122 - tmp104
    tmp124 = tl.load(in_ptr29 + (x1), tmp101, eviction_policy='evict_last', other=0.0)
    tmp125 = tmp123 * tmp124
    tmp126 = tmp104 + tmp125
    tmp127 = tl.full(tmp126.shape, 0.0, tmp126.dtype)
    tmp128 = tl.where(tmp101, tmp126, tmp127)
    tmp129 = tl.where(tmp74, tmp100, tmp128)
    tmp130 = tl.where(tmp44, tmp70, tmp129)
    tmp131 = tl.where(tmp14, tmp40, tmp130)
    tmp132 = tl.where(tmp9, tmp10, tmp131)
    tmp133 = tl.where(tmp4, tmp5, tmp132)
    tl.store(out_ptr0 + (x5), tmp133, None)
