
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*i64', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*i64', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*i64', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*i64', 'in_ptr23': '*i64', 'in_ptr24': '*fp32', 'in_ptr25': '*i64', 'in_ptr26': '*fp32', 'in_ptr27': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 416)
    x3 = xindex // 26624
    x4 = ((xindex // 416) % 64)
    x2 = ((xindex // 3328) % 8)
    x1 = ((xindex // 416) % 8)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 64*(x0) + 4096*x3), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 32, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = tmp11 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp11)
    tmp15 = tl.load(in_ptr3 + (64*tmp14 + 2048*tmp10 + 65536*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp7
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr3 + (64*tmp19 + 2048*tmp10 + 65536*x3 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 - tmp15
    tmp22 = tl.load(in_ptr5 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp15 + tmp23
    tmp25 = tmp24 - tmp5
    tmp26 = tl.load(in_ptr6 + (x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp5 + tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp4, tmp28, tmp29)
    tmp31 = tmp0 >= tmp3
    tmp32 = tl.full([1], 160, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr7 + (x4 + 64*((-64) + x0) + 6144*x3), tmp34, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr8 + (x2), tmp34, eviction_policy='evict_last', other=0.0)
    tmp37 = tl.full([XBLOCK], 16, tl.int32)
    tmp38 = tmp36 + tmp37
    tmp39 = tmp36 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp36)
    tmp41 = tl.load(in_ptr9 + (x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp41 + tmp37
    tmp43 = tmp41 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp41)
    tmp45 = tl.load(in_ptr10 + (96*tmp44 + 1536*tmp40 + 24576*x3 + ((-64) + x0)), tmp34, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr11 + (x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp47 = tmp46 + tmp37
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tmp50 = tl.load(in_ptr10 + (96*tmp49 + 1536*tmp40 + 24576*x3 + ((-64) + x0)), tmp34, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 - tmp45
    tmp52 = tl.load(in_ptr12 + (x1), tmp34, eviction_policy='evict_last', other=0.0)
    tmp53 = tmp51 * tmp52
    tmp54 = tmp45 + tmp53
    tmp55 = tmp54 - tmp35
    tmp56 = tl.load(in_ptr13 + (x2), tmp34, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tmp35 + tmp57
    tmp59 = tl.full(tmp58.shape, 0.0, tmp58.dtype)
    tmp60 = tl.where(tmp34, tmp58, tmp59)
    tmp61 = tmp0 >= tmp32
    tmp62 = tl.full([1], 288, tl.int64)
    tmp63 = tmp0 < tmp62
    tmp64 = tmp61 & tmp63
    tmp65 = tl.load(in_ptr14 + (x4 + 64*((-160) + x0) + 8192*x3), tmp64, eviction_policy='evict_last', other=0.0)
    tmp66 = tl.load(in_ptr15 + (x2), tmp64, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full([XBLOCK], 8, tl.int32)
    tmp68 = tmp66 + tmp67
    tmp69 = tmp66 < 0
    tmp70 = tl.where(tmp69, tmp68, tmp66)
    tmp71 = tl.load(in_ptr16 + (x1), tmp64, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp71 + tmp67
    tmp73 = tmp71 < 0
    tmp74 = tl.where(tmp73, tmp72, tmp71)
    tmp75 = tl.load(in_ptr17 + (128*tmp74 + 1024*tmp70 + 8192*x3 + ((-160) + x0)), tmp64, eviction_policy='evict_last', other=0.0)
    tmp76 = tl.load(in_ptr18 + (x1), tmp64, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp76 + tmp67
    tmp78 = tmp76 < 0
    tmp79 = tl.where(tmp78, tmp77, tmp76)
    tmp80 = tl.load(in_ptr17 + (128*tmp79 + 1024*tmp70 + 8192*x3 + ((-160) + x0)), tmp64, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 - tmp75
    tmp82 = tl.load(in_ptr19 + (x1), tmp64, eviction_policy='evict_last', other=0.0)
    tmp83 = tmp81 * tmp82
    tmp84 = tmp75 + tmp83
    tmp85 = tmp84 - tmp65
    tmp86 = tl.load(in_ptr20 + (x2), tmp64, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 * tmp86
    tmp88 = tmp65 + tmp87
    tmp89 = tl.full(tmp88.shape, 0.0, tmp88.dtype)
    tmp90 = tl.where(tmp64, tmp88, tmp89)
    tmp91 = tmp0 >= tmp62
    tmp92 = tl.full([1], 416, tl.int64)
    tmp93 = tmp0 < tmp92
    tmp94 = tl.load(in_ptr21 + (x4 + 64*((-288) + x0) + 8192*x3), tmp91, eviction_policy='evict_last', other=0.0)
    tmp95 = tl.load(in_ptr22 + (x2), tmp91, eviction_policy='evict_last', other=0.0)
    tmp96 = tl.full([XBLOCK], 4, tl.int32)
    tmp97 = tmp95 + tmp96
    tmp98 = tmp95 < 0
    tmp99 = tl.where(tmp98, tmp97, tmp95)
    tmp100 = tl.load(in_ptr23 + (x1), tmp91, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp100 + tmp96
    tmp102 = tmp100 < 0
    tmp103 = tl.where(tmp102, tmp101, tmp100)
    tmp104 = tl.load(in_ptr24 + (128*tmp103 + 512*tmp99 + 2048*x3 + ((-288) + x0)), tmp91, eviction_policy='evict_last', other=0.0)
    tmp105 = tl.load(in_ptr25 + (x1), tmp91, eviction_policy='evict_last', other=0.0)
    tmp106 = tmp105 + tmp96
    tmp107 = tmp105 < 0
    tmp108 = tl.where(tmp107, tmp106, tmp105)
    tmp109 = tl.load(in_ptr24 + (128*tmp108 + 512*tmp99 + 2048*x3 + ((-288) + x0)), tmp91, eviction_policy='evict_last', other=0.0)
    tmp110 = tmp109 - tmp104
    tmp111 = tl.load(in_ptr26 + (x1), tmp91, eviction_policy='evict_last', other=0.0)
    tmp112 = tmp110 * tmp111
    tmp113 = tmp104 + tmp112
    tmp114 = tmp113 - tmp94
    tmp115 = tl.load(in_ptr27 + (x2), tmp91, eviction_policy='evict_last', other=0.0)
    tmp116 = tmp114 * tmp115
    tmp117 = tmp94 + tmp116
    tmp118 = tl.full(tmp117.shape, 0.0, tmp117.dtype)
    tmp119 = tl.where(tmp91, tmp117, tmp118)
    tmp120 = tl.where(tmp64, tmp90, tmp119)
    tmp121 = tl.where(tmp34, tmp60, tmp120)
    tmp122 = tl.where(tmp4, tmp30, tmp121)
    tl.store(out_ptr0 + (x5), tmp122, None)
