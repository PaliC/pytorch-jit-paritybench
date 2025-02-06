
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*i64', 'in_ptr18': '*fp32', 'in_ptr19': '*i64', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*i64', 'in_ptr24': '*i64', 'in_ptr25': '*fp32', 'in_ptr26': '*i64', 'in_ptr27': '*fp32', 'in_ptr28': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 26, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 16) % 4096)
    x3 = xindex // 65536
    x4 = (xindex % 16)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 32768*x3), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2560, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x4 + 16*((-2048) + x2) + 8192*x3), tmp9, other=0.0)
    tmp11 = tl.load(in_ptr2 + (x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([XBLOCK], 1, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tmp16 = tl.load(in_ptr3 + (x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr4 + (512*x3 + ((-2048) + x2)), tmp9, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp12
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tmp20 - tmp20
    tmp26 = tl.load(in_ptr6 + (x0), tmp9, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 + tmp27
    tmp29 = tmp28 - tmp10
    tmp30 = tl.load(in_ptr7 + (x1), tmp9, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp10 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp9, tmp32, tmp33)
    tmp35 = tmp0 >= tmp7
    tmp36 = tl.full([1], 3072, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tmp35 & tmp37
    tmp39 = tl.load(in_ptr8 + (x4 + 16*((-2560) + x2) + 8192*x3), tmp38, other=0.0)
    tmp40 = tl.load(in_ptr9 + (x1), tmp38, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full([XBLOCK], 2, tl.int32)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp40 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp40)
    tmp45 = tl.load(in_ptr10 + (x0), tmp38, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tmp49 = tl.load(in_ptr11 + (512*tmp48 + 1024*tmp44 + 2048*x3 + ((-2560) + x2)), tmp38, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr12 + (x0), tmp38, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp41
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tl.load(in_ptr11 + (512*tmp53 + 1024*tmp44 + 2048*x3 + ((-2560) + x2)), tmp38, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 - tmp49
    tmp56 = tl.load(in_ptr13 + (x0), tmp38, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tmp49 + tmp57
    tmp59 = tmp58 - tmp39
    tmp60 = tl.load(in_ptr14 + (x1), tmp38, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp59 * tmp60
    tmp62 = tmp39 + tmp61
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp38, tmp62, tmp63)
    tmp65 = tmp0 >= tmp36
    tmp66 = tl.full([1], 3584, tl.int64)
    tmp67 = tmp0 < tmp66
    tmp68 = tmp65 & tmp67
    tmp69 = tl.load(in_ptr15 + (x4 + 16*((-3072) + x2) + 8192*x3), tmp68, other=0.0)
    tmp70 = tl.load(in_ptr16 + (x1), tmp68, eviction_policy='evict_last', other=0.0)
    tmp71 = tl.full([XBLOCK], 3, tl.int32)
    tmp72 = tmp70 + tmp71
    tmp73 = tmp70 < 0
    tmp74 = tl.where(tmp73, tmp72, tmp70)
    tmp75 = tl.load(in_ptr17 + (x0), tmp68, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp75 + tmp71
    tmp77 = tmp75 < 0
    tmp78 = tl.where(tmp77, tmp76, tmp75)
    tmp79 = tl.load(in_ptr18 + (512*tmp78 + 1536*tmp74 + 4608*x3 + ((-3072) + x2)), tmp68, eviction_policy='evict_last', other=0.0)
    tmp80 = tl.load(in_ptr19 + (x0), tmp68, eviction_policy='evict_last', other=0.0)
    tmp81 = tmp80 + tmp71
    tmp82 = tmp80 < 0
    tmp83 = tl.where(tmp82, tmp81, tmp80)
    tmp84 = tl.load(in_ptr18 + (512*tmp83 + 1536*tmp74 + 4608*x3 + ((-3072) + x2)), tmp68, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp84 - tmp79
    tmp86 = tl.load(in_ptr20 + (x0), tmp68, eviction_policy='evict_last', other=0.0)
    tmp87 = tmp85 * tmp86
    tmp88 = tmp79 + tmp87
    tmp89 = tmp88 - tmp69
    tmp90 = tl.load(in_ptr21 + (x1), tmp68, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 * tmp90
    tmp92 = tmp69 + tmp91
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp68, tmp92, tmp93)
    tmp95 = tmp0 >= tmp66
    tmp96 = tl.full([1], 4096, tl.int64)
    tmp97 = tmp0 < tmp96
    tmp98 = tl.load(in_ptr22 + (x4 + 16*((-3584) + x2) + 8192*x3), tmp95, other=0.0)
    tmp99 = tl.load(in_ptr23 + (x1), tmp95, eviction_policy='evict_last', other=0.0)
    tmp100 = tl.full([XBLOCK], 6, tl.int32)
    tmp101 = tmp99 + tmp100
    tmp102 = tmp99 < 0
    tmp103 = tl.where(tmp102, tmp101, tmp99)
    tmp104 = tl.load(in_ptr24 + (x0), tmp95, eviction_policy='evict_last', other=0.0)
    tmp105 = tmp104 + tmp100
    tmp106 = tmp104 < 0
    tmp107 = tl.where(tmp106, tmp105, tmp104)
    tmp108 = tl.load(in_ptr25 + (512*tmp107 + 3072*tmp103 + 18432*x3 + ((-3584) + x2)), tmp95, eviction_policy='evict_last', other=0.0)
    tmp109 = tl.load(in_ptr26 + (x0), tmp95, eviction_policy='evict_last', other=0.0)
    tmp110 = tmp109 + tmp100
    tmp111 = tmp109 < 0
    tmp112 = tl.where(tmp111, tmp110, tmp109)
    tmp113 = tl.load(in_ptr25 + (512*tmp112 + 3072*tmp103 + 18432*x3 + ((-3584) + x2)), tmp95, eviction_policy='evict_last', other=0.0)
    tmp114 = tmp113 - tmp108
    tmp115 = tl.load(in_ptr27 + (x0), tmp95, eviction_policy='evict_last', other=0.0)
    tmp116 = tmp114 * tmp115
    tmp117 = tmp108 + tmp116
    tmp118 = tmp117 - tmp98
    tmp119 = tl.load(in_ptr28 + (x1), tmp95, eviction_policy='evict_last', other=0.0)
    tmp120 = tmp118 * tmp119
    tmp121 = tmp98 + tmp120
    tmp122 = tl.full(tmp121.shape, 0.0, tmp121.dtype)
    tmp123 = tl.where(tmp95, tmp121, tmp122)
    tmp124 = tl.where(tmp68, tmp94, tmp123)
    tmp125 = tl.where(tmp38, tmp64, tmp124)
    tmp126 = tl.where(tmp9, tmp34, tmp125)
    tmp127 = tl.where(tmp4, tmp5, tmp126)
    tl.store(out_ptr0 + (x5), tmp127, None)
