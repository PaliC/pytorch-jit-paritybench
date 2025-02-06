
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*i64', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*i64', 'in_ptr23': '*i64', 'in_ptr24': '*fp32', 'in_ptr25': '*i64', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 1, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (4*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tmp15 - tmp15
    tmp21 = tl.load(in_ptr5 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp20 * tmp21
    tmp23 = tmp15 + tmp22
    tmp24 = tmp23 - tmp5
    tmp25 = tl.load(in_ptr6 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp5 + tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp4, tmp27, tmp28)
    tmp30 = tmp0 >= tmp3
    tmp31 = tl.full([1], 8, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tmp30 & tmp32
    tmp34 = tl.load(in_ptr7 + (x4 + 16*((-4) + x0) + 64*x3), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr8 + (x2), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.full([XBLOCK], 2, tl.int32)
    tmp37 = tmp35 + tmp36
    tmp38 = tmp35 < 0
    tmp39 = tl.where(tmp38, tmp37, tmp35)
    tmp40 = tl.load(in_ptr9 + (x1), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp40 + tmp36
    tmp42 = tmp40 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp40)
    tmp44 = tl.load(in_ptr10 + (4*tmp43 + 8*tmp39 + 16*x3 + ((-4) + x0)), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp45 = tl.load(in_ptr11 + (x1), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp36
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tmp49 = tl.load(in_ptr10 + (4*tmp48 + 8*tmp39 + 16*x3 + ((-4) + x0)), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 - tmp44
    tmp51 = tl.load(in_ptr12 + (x1), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp44 + tmp52
    tmp54 = tmp53 - tmp34
    tmp55 = tl.load(in_ptr13 + (x2), tmp33 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp34 + tmp56
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp33, tmp57, tmp58)
    tmp60 = tmp0 >= tmp31
    tmp61 = tl.full([1], 12, tl.int64)
    tmp62 = tmp0 < tmp61
    tmp63 = tmp60 & tmp62
    tmp64 = tl.load(in_ptr14 + (x4 + 16*((-8) + x0) + 64*x3), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp65 = tl.load(in_ptr15 + (x2), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp66 = tl.full([XBLOCK], 3, tl.int32)
    tmp67 = tmp65 + tmp66
    tmp68 = tmp65 < 0
    tmp69 = tl.where(tmp68, tmp67, tmp65)
    tmp70 = tl.load(in_ptr16 + (x1), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp71 = tmp70 + tmp66
    tmp72 = tmp70 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp70)
    tmp74 = tl.load(in_ptr17 + (4*tmp73 + 12*tmp69 + 36*x3 + ((-8) + x0)), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tl.load(in_ptr18 + (x1), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp75 + tmp66
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tmp79 = tl.load(in_ptr17 + (4*tmp78 + 12*tmp69 + 36*x3 + ((-8) + x0)), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tmp79 - tmp74
    tmp81 = tl.load(in_ptr19 + (x1), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp82 = tmp80 * tmp81
    tmp83 = tmp74 + tmp82
    tmp84 = tmp83 - tmp64
    tmp85 = tl.load(in_ptr20 + (x2), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp84 * tmp85
    tmp87 = tmp64 + tmp86
    tmp88 = tl.full(tmp87.shape, 0.0, tmp87.dtype)
    tmp89 = tl.where(tmp63, tmp87, tmp88)
    tmp90 = tmp0 >= tmp61
    tmp91 = tl.full([1], 16, tl.int64)
    tmp92 = tmp0 < tmp91
    tmp93 = tmp90 & tmp92
    tmp94 = tl.load(in_ptr21 + (x4 + 16*((-12) + x0) + 64*x3), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp95 = tl.load(in_ptr22 + (x2), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp96 = tl.full([XBLOCK], 6, tl.int32)
    tmp97 = tmp95 + tmp96
    tmp98 = tmp95 < 0
    tmp99 = tl.where(tmp98, tmp97, tmp95)
    tmp100 = tl.load(in_ptr23 + (x1), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp100 + tmp96
    tmp102 = tmp100 < 0
    tmp103 = tl.where(tmp102, tmp101, tmp100)
    tmp104 = tl.load(in_ptr24 + (4*tmp103 + 24*tmp99 + 144*x3 + ((-12) + x0)), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp105 = tl.load(in_ptr25 + (x1), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp106 = tmp105 + tmp96
    tmp107 = tmp105 < 0
    tmp108 = tl.where(tmp107, tmp106, tmp105)
    tmp109 = tl.load(in_ptr24 + (4*tmp108 + 24*tmp99 + 144*x3 + ((-12) + x0)), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp110 = tmp109 - tmp104
    tmp111 = tl.load(in_ptr26 + (x1), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp112 = tmp110 * tmp111
    tmp113 = tmp104 + tmp112
    tmp114 = tmp113 - tmp94
    tmp115 = tl.load(in_ptr27 + (x2), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp116 = tmp114 * tmp115
    tmp117 = tmp94 + tmp116
    tmp118 = tl.full(tmp117.shape, 0.0, tmp117.dtype)
    tmp119 = tl.where(tmp93, tmp117, tmp118)
    tmp120 = tmp0 >= tmp91
    tmp121 = tl.full([1], 20, tl.int64)
    tmp122 = tmp0 < tmp121
    tmp123 = tl.load(in_ptr28 + (x4 + 16*((-16) + x0) + 64*x3), tmp120 & xmask, eviction_policy='evict_last', other=0.0)
    tmp124 = tl.where(tmp93, tmp119, tmp123)
    tmp125 = tl.where(tmp63, tmp89, tmp124)
    tmp126 = tl.where(tmp33, tmp59, tmp125)
    tmp127 = tl.where(tmp4, tmp29, tmp126)
    tl.store(out_ptr0 + (x5), tmp127, xmask)
