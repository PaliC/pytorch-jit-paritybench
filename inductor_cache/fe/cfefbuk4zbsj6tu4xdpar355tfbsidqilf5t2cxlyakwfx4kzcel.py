
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 28, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 20)
    x3 = xindex // 320
    x4 = ((xindex // 20) % 16)
    x2 = ((xindex // 80) % 4)
    x1 = ((xindex // 20) % 4)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x0) + 64*x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 16*((-4) + x0) + 64*x3), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.load(in_ptr2 + (x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([XBLOCK], 2, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tmp16 = tl.load(in_ptr3 + (x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr4 + (4*tmp19 + 8*tmp15 + 16*x3 + ((-4) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp12
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tl.load(in_ptr4 + (4*tmp24 + 8*tmp15 + 16*x3 + ((-4) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 - tmp20
    tmp27 = tl.load(in_ptr6 + (x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 * tmp27
    tmp29 = tmp20 + tmp28
    tmp30 = tmp29 - tmp10
    tmp31 = tl.load(in_ptr7 + (x2), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp10 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp9, tmp33, tmp34)
    tmp36 = tmp0 >= tmp7
    tmp37 = tl.full([1], 12, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp36 & tmp38
    tmp40 = tl.load(in_ptr8 + (x4 + 16*((-8) + x0) + 64*x3), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr9 + (x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp42 = tl.full([XBLOCK], 1, tl.int32)
    tmp43 = tmp41 + tmp42
    tmp44 = tmp41 < 0
    tmp45 = tl.where(tmp44, tmp43, tmp41)
    tmp46 = tl.load(in_ptr10 + (x1), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp46 + tmp42
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tmp50 = tl.load(in_ptr11 + (4*x3 + ((-8) + x0)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tl.load(in_ptr12 + (x1), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp42
    tmp53 = tmp51 < 0
    tmp54 = tl.where(tmp53, tmp52, tmp51)
    tmp55 = tmp50 - tmp50
    tmp56 = tl.load(in_ptr13 + (x1), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tmp50 + tmp57
    tmp59 = tmp58 - tmp40
    tmp60 = tl.load(in_ptr14 + (x2), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp40 + tmp61
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp39, tmp62, tmp63)
    tmp65 = tmp0 >= tmp37
    tmp66 = tl.full([1], 16, tl.int64)
    tmp67 = tmp0 < tmp66
    tmp68 = tmp65 & tmp67
    tmp69 = tl.load(in_ptr15 + (x4 + 16*((-12) + x0) + 64*x3), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tl.load(in_ptr9 + (x2), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.full([XBLOCK], 1, tl.int32)
    tmp72 = tmp70 + tmp71
    tmp73 = tmp70 < 0
    tmp74 = tl.where(tmp73, tmp72, tmp70)
    tmp75 = tl.load(in_ptr10 + (x1), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp75 + tmp71
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tmp79 = tl.load(in_ptr16 + (4*x3 + ((-12) + x0)), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tl.load(in_ptr12 + (x1), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp71
    tmp82 = tmp80 < 0
    tmp83 = tl.where(tmp82, tmp81, tmp80)
    tmp84 = tmp79 - tmp79
    tmp85 = tl.load(in_ptr13 + (x1), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp84 * tmp85
    tmp87 = tmp79 + tmp86
    tmp88 = tmp87 - tmp69
    tmp89 = tl.load(in_ptr14 + (x2), tmp68 & xmask, eviction_policy='evict_last', other=0.0)
    tmp90 = tmp88 * tmp89
    tmp91 = tmp69 + tmp90
    tmp92 = tl.full(tmp91.shape, 0.0, tmp91.dtype)
    tmp93 = tl.where(tmp68, tmp91, tmp92)
    tmp94 = tmp0 >= tmp66
    tmp95 = tl.full([1], 20, tl.int64)
    tmp96 = tmp0 < tmp95
    tmp97 = tl.load(in_ptr17 + (x4 + 16*((-16) + x0) + 64*x3), tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp98 = tl.load(in_ptr9 + (x2), tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp99 = tl.full([XBLOCK], 1, tl.int32)
    tmp100 = tmp98 + tmp99
    tmp101 = tmp98 < 0
    tmp102 = tl.where(tmp101, tmp100, tmp98)
    tmp103 = tl.load(in_ptr10 + (x1), tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp104 = tmp103 + tmp99
    tmp105 = tmp103 < 0
    tmp106 = tl.where(tmp105, tmp104, tmp103)
    tmp107 = tl.load(in_ptr18 + (4*x3 + ((-16) + x0)), tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tl.load(in_ptr12 + (x1), tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp109 = tmp108 + tmp99
    tmp110 = tmp108 < 0
    tmp111 = tl.where(tmp110, tmp109, tmp108)
    tmp112 = tmp107 - tmp107
    tmp113 = tl.load(in_ptr13 + (x1), tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp114 = tmp112 * tmp113
    tmp115 = tmp107 + tmp114
    tmp116 = tmp115 - tmp97
    tmp117 = tl.load(in_ptr14 + (x2), tmp94 & xmask, eviction_policy='evict_last', other=0.0)
    tmp118 = tmp116 * tmp117
    tmp119 = tmp97 + tmp118
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp94, tmp119, tmp120)
    tmp122 = tl.where(tmp68, tmp93, tmp121)
    tmp123 = tl.where(tmp39, tmp64, tmp122)
    tmp124 = tl.where(tmp9, tmp35, tmp123)
    tmp125 = tl.where(tmp4, tmp5, tmp124)
    tl.store(out_ptr0 + (x5), tmp125, xmask)
