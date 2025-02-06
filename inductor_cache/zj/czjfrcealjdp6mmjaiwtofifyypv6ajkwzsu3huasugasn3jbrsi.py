
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 29, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 16) % 2052)
    x3 = xindex // 32832
    x4 = (xindex % 16)
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x7 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 16*(x2) + 64*x3), tmp4 & xmask, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 516, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (33 + x0 + 10*x1 + 100*((-4) + x2) + 51200*x3), tmp9 & xmask, other=0.0)
    tmp11 = tl.load(in_ptr2 + (3 + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full([XBLOCK], 1, tl.int32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp11 < 0
    tmp15 = tl.where(tmp14, tmp13, tmp11)
    tmp16 = tl.load(in_ptr3 + (3 + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp16 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp16)
    tmp20 = tl.load(in_ptr4 + (512*x3 + ((-4) + x2)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.load(in_ptr5 + (3 + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = tmp21 + tmp12
    tmp23 = tmp21 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp21)
    tmp25 = tmp20 - tmp20
    tmp26 = tl.load(in_ptr6 + (3 + x0), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 + tmp27
    tmp29 = tmp28 - tmp10
    tmp30 = tl.load(in_ptr7 + (3 + x1), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp10 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp9, tmp32, tmp33)
    tmp35 = tmp0 >= tmp7
    tmp36 = tl.full([1], 1028, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tmp35 & tmp37
    tmp39 = tl.load(in_ptr8 + (168 + x0 + 20*x1 + 400*((-516) + x2) + 204800*x3), tmp38 & xmask, other=0.0)
    tmp40 = tl.load(in_ptr9 + (8 + x1), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.full([XBLOCK], 1, tl.int32)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp40 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp40)
    tmp45 = tl.load(in_ptr10 + (8 + x0), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp45 < 0
    tmp48 = tl.where(tmp47, tmp46, tmp45)
    tmp49 = tl.load(in_ptr11 + (512*x3 + ((-516) + x2)), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.load(in_ptr12 + (8 + x0), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp41
    tmp52 = tmp50 < 0
    tmp53 = tl.where(tmp52, tmp51, tmp50)
    tmp54 = tmp49 - tmp49
    tmp55 = tl.load(in_ptr13 + (8 + x0), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp56 = tmp54 * tmp55
    tmp57 = tmp49 + tmp56
    tmp58 = tmp57 - tmp39
    tmp59 = tl.load(in_ptr14 + (8 + x1), tmp38 & xmask, eviction_policy='evict_last', other=0.0)
    tmp60 = tmp58 * tmp59
    tmp61 = tmp39 + tmp60
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp38, tmp61, tmp62)
    tmp64 = tmp0 >= tmp36
    tmp65 = tl.full([1], 1540, tl.int64)
    tmp66 = tmp0 < tmp65
    tmp67 = tmp64 & tmp66
    tmp68 = tl.load(in_ptr15 + (403 + x0 + 30*x1 + 900*((-1028) + x2) + 460800*x3), tmp67 & xmask, other=0.0)
    tmp69 = tl.load(in_ptr16 + (13 + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tl.full([XBLOCK], 1, tl.int32)
    tmp71 = tmp69 + tmp70
    tmp72 = tmp69 < 0
    tmp73 = tl.where(tmp72, tmp71, tmp69)
    tmp74 = tl.load(in_ptr17 + (13 + x0), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp75 = tmp74 + tmp70
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tmp78 = tl.load(in_ptr18 + (512*x3 + ((-1028) + x2)), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp79 = tl.load(in_ptr19 + (13 + x0), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp80 = tmp79 + tmp70
    tmp81 = tmp79 < 0
    tmp82 = tl.where(tmp81, tmp80, tmp79)
    tmp83 = tmp78 - tmp78
    tmp84 = tl.load(in_ptr20 + (13 + x0), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp85 = tmp83 * tmp84
    tmp86 = tmp78 + tmp85
    tmp87 = tmp86 - tmp68
    tmp88 = tl.load(in_ptr21 + (13 + x1), tmp67 & xmask, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tmp68 + tmp89
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp67, tmp90, tmp91)
    tmp93 = tmp0 >= tmp65
    tmp94 = tl.full([1], 2052, tl.int64)
    tmp95 = tmp0 < tmp94
    tmp96 = tl.load(in_ptr22 + (1708 + x0 + 60*x1 + 3616*((-1540) + x2) + 1851392*x3), tmp93 & xmask, other=0.0)
    tmp97 = tl.load(in_ptr23 + (28 + x1), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp98 = tl.full([XBLOCK], 1, tl.int32)
    tmp99 = tmp97 + tmp98
    tmp100 = tmp97 < 0
    tmp101 = tl.where(tmp100, tmp99, tmp97)
    tmp102 = tl.load(in_ptr24 + (28 + x0), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp103 = tmp102 + tmp98
    tmp104 = tmp102 < 0
    tmp105 = tl.where(tmp104, tmp103, tmp102)
    tmp106 = tl.load(in_ptr25 + (512*x3 + ((-1540) + x2)), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp107 = tl.load(in_ptr26 + (28 + x0), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp108 = tmp107 + tmp98
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tmp106 - tmp106
    tmp112 = tl.load(in_ptr27 + (28 + x0), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp113 = tmp111 * tmp112
    tmp114 = tmp106 + tmp113
    tmp115 = tmp114 - tmp96
    tmp116 = tl.load(in_ptr28 + (28 + x1), tmp93 & xmask, eviction_policy='evict_last', other=0.0)
    tmp117 = tmp115 * tmp116
    tmp118 = tmp96 + tmp117
    tmp119 = tl.full(tmp118.shape, 0.0, tmp118.dtype)
    tmp120 = tl.where(tmp93, tmp118, tmp119)
    tmp121 = tl.where(tmp67, tmp92, tmp120)
    tmp122 = tl.where(tmp38, tmp63, tmp121)
    tmp123 = tl.where(tmp9, tmp34, tmp122)
    tmp124 = tl.where(tmp4, tmp5, tmp123)
    tl.store(out_ptr0 + (x7), tmp124, xmask)
