
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*i64', 'in_ptr22': '*i64', 'in_ptr23': '*fp32', 'in_ptr24': '*i64', 'in_ptr25': '*fp32', 'in_ptr26': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 27, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1280)
    x4 = xindex // 1280
    x3 = xindex // 20480
    x5 = ((xindex // 1280) % 16)
    x2 = ((xindex // 5120) % 4)
    x1 = ((xindex // 1280) % 4)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x4 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 512, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (256*x4 + ((-256) + x0)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-256) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-256) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-256) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-256) + x0), tmp28, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 768, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (256*x4 + ((-512) + x0)), tmp52, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-512) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-512) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp57 = 1e-05
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-512) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-512) + x0), tmp52, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 1024, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tmp73 & tmp75
    tmp77 = tl.load(in_ptr15 + (256*x4 + ((-768) + x0)), tmp76, eviction_policy='evict_last', other=0.0)
    tmp78 = tl.load(in_ptr16 + ((-768) + x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp79 = tmp77 - tmp78
    tmp80 = tl.load(in_ptr17 + ((-768) + x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp81 = 1e-05
    tmp82 = tmp80 + tmp81
    tmp83 = libdevice.sqrt(tmp82)
    tmp84 = tl.full([1], 1, tl.int32)
    tmp85 = tmp84 / tmp83
    tmp86 = 1.0
    tmp87 = tmp85 * tmp86
    tmp88 = tmp79 * tmp87
    tmp89 = tl.load(in_ptr18 + ((-768) + x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp90 = tmp88 * tmp89
    tmp91 = tl.load(in_ptr19 + ((-768) + x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp92 = tmp90 + tmp91
    tmp93 = tl.full([1], 0, tl.int32)
    tmp94 = triton_helpers.maximum(tmp93, tmp92)
    tmp95 = tl.full(tmp94.shape, 0.0, tmp94.dtype)
    tmp96 = tl.where(tmp76, tmp94, tmp95)
    tmp97 = tmp0 >= tmp74
    tmp98 = tl.full([1], 1280, tl.int64)
    tmp99 = tmp0 < tmp98
    tmp100 = tl.load(in_ptr20 + (x5 + 16*((-1024) + x0) + 4096*x3), tmp97, eviction_policy='evict_last', other=0.0)
    tmp101 = tl.load(in_ptr21 + (x2), tmp97, eviction_policy='evict_last', other=0.0)
    tmp102 = tl.full([XBLOCK], 1, tl.int32)
    tmp103 = tmp101 + tmp102
    tmp104 = tmp101 < 0
    tmp105 = tl.where(tmp104, tmp103, tmp101)
    tmp106 = tl.load(in_ptr22 + (x1), tmp97, eviction_policy='evict_last', other=0.0)
    tmp107 = tmp106 + tmp102
    tmp108 = tmp106 < 0
    tmp109 = tl.where(tmp108, tmp107, tmp106)
    tmp110 = tl.load(in_ptr23 + (256*x3 + ((-1024) + x0)), tmp97, eviction_policy='evict_last', other=0.0)
    tmp111 = tl.load(in_ptr24 + (x1), tmp97, eviction_policy='evict_last', other=0.0)
    tmp112 = tmp111 + tmp102
    tmp113 = tmp111 < 0
    tmp114 = tl.where(tmp113, tmp112, tmp111)
    tmp115 = tmp110 - tmp110
    tmp116 = tl.load(in_ptr25 + (x1), tmp97, eviction_policy='evict_last', other=0.0)
    tmp117 = tmp115 * tmp116
    tmp118 = tmp110 + tmp117
    tmp119 = tmp118 - tmp100
    tmp120 = tl.load(in_ptr26 + (x2), tmp97, eviction_policy='evict_last', other=0.0)
    tmp121 = tmp119 * tmp120
    tmp122 = tmp100 + tmp121
    tmp123 = tl.full(tmp122.shape, 0.0, tmp122.dtype)
    tmp124 = tl.where(tmp97, tmp122, tmp123)
    tmp125 = tl.where(tmp76, tmp96, tmp124)
    tmp126 = tl.where(tmp52, tmp72, tmp125)
    tmp127 = tl.where(tmp28, tmp48, tmp126)
    tmp128 = tl.where(tmp4, tmp24, tmp127)
    tl.store(out_ptr0 + (x6), tmp128, None)
