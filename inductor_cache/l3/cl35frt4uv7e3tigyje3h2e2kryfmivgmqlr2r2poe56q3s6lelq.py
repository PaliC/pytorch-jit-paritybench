
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*fp32', 'in_ptr11': '*i64', 'in_ptr12': '*fp32', 'in_ptr13': '*i64', 'in_ptr14': '*i64', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*fp32', 'in_ptr18': '*i64', 'in_ptr19': '*fp32', 'in_ptr20': '*i64', 'in_ptr21': '*i64', 'in_ptr22': '*fp32', 'in_ptr23': '*i64', 'in_ptr24': '*fp32', 'in_ptr25': '*i64', 'in_ptr26': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training__unsafe_index_add_mul_relu_sub_threshold_backward_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, out_ptr0, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x6 = xindex
    x1 = ((xindex // 256) % 18)
    x4 = ((xindex // 16) % 16)
    x3 = (xindex % 16)
    x5 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x6), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x6), xmask)
    tmp20 = tl.load(in_ptr6 + (x4), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x3), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr9 + (x3), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr10 + (x3), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr11 + (x4), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr12 + (x4), xmask, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr13 + (x4), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr14 + (x3), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr16 + (x3), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr17 + (x3), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr18 + (x4), xmask, eviction_policy='evict_last')
    tmp84 = tl.load(in_ptr19 + (x4), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr20 + (x4), xmask, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr21 + (x3), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr23 + (x3), xmask, eviction_policy='evict_last')
    tmp104 = tl.load(in_ptr24 + (x3), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr25 + (x4), xmask, eviction_policy='evict_last')
    tmp117 = tl.load(in_ptr26 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp21 = tl.full([XBLOCK], 8, tl.int32)
    tmp22 = tmp20 + tmp21
    tmp23 = tmp20 < 0
    tmp24 = tl.where(tmp23, tmp22, tmp20)
    tmp26 = tmp25 + tmp21
    tmp27 = tmp25 < 0
    tmp28 = tl.where(tmp27, tmp26, tmp25)
    tmp29 = tl.load(in_ptr8 + (tmp28 + 8*tmp24 + 64*x5), xmask, eviction_policy='evict_last')
    tmp31 = tmp30 + tmp21
    tmp32 = tmp30 < 0
    tmp33 = tl.where(tmp32, tmp31, tmp30)
    tmp34 = tl.load(in_ptr8 + (tmp33 + 8*tmp24 + 64*x5), xmask, eviction_policy='evict_last')
    tmp35 = tmp34 - tmp29
    tmp37 = tmp35 * tmp36
    tmp38 = tmp29 + tmp37
    tmp39 = 0.0
    tmp40 = tmp19 + tmp39
    tmp42 = tmp41 + tmp21
    tmp43 = tmp41 < 0
    tmp44 = tl.where(tmp43, tmp42, tmp41)
    tmp45 = tl.load(in_ptr8 + (tmp28 + 8*tmp44 + 64*x5), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (tmp33 + 8*tmp44 + 64*x5), xmask, eviction_policy='evict_last')
    tmp47 = tmp46 - tmp45
    tmp48 = tmp47 * tmp36
    tmp49 = tmp45 + tmp48
    tmp50 = tmp49 - tmp38
    tmp52 = tmp50 * tmp51
    tmp53 = tmp38 + tmp52
    tmp54 = tmp40 + tmp53
    tmp56 = tl.full([XBLOCK], 4, tl.int32)
    tmp57 = tmp55 + tmp56
    tmp58 = tmp55 < 0
    tmp59 = tl.where(tmp58, tmp57, tmp55)
    tmp61 = tmp60 + tmp56
    tmp62 = tmp60 < 0
    tmp63 = tl.where(tmp62, tmp61, tmp60)
    tmp64 = tl.load(in_ptr15 + (tmp63 + 4*tmp59 + 16*x5), xmask, eviction_policy='evict_last')
    tmp66 = tmp65 + tmp56
    tmp67 = tmp65 < 0
    tmp68 = tl.where(tmp67, tmp66, tmp65)
    tmp69 = tl.load(in_ptr15 + (tmp68 + 4*tmp59 + 16*x5), xmask, eviction_policy='evict_last')
    tmp70 = tmp69 - tmp64
    tmp72 = tmp70 * tmp71
    tmp73 = tmp64 + tmp72
    tmp75 = tmp74 + tmp56
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tmp78 = tl.load(in_ptr15 + (tmp63 + 4*tmp77 + 16*x5), xmask, eviction_policy='evict_last')
    tmp79 = tl.load(in_ptr15 + (tmp68 + 4*tmp77 + 16*x5), xmask, eviction_policy='evict_last')
    tmp80 = tmp79 - tmp78
    tmp81 = tmp80 * tmp71
    tmp82 = tmp78 + tmp81
    tmp83 = tmp82 - tmp73
    tmp85 = tmp83 * tmp84
    tmp86 = tmp73 + tmp85
    tmp87 = tmp54 + tmp86
    tmp89 = tl.full([XBLOCK], 2, tl.int32)
    tmp90 = tmp88 + tmp89
    tmp91 = tmp88 < 0
    tmp92 = tl.where(tmp91, tmp90, tmp88)
    tmp94 = tmp93 + tmp89
    tmp95 = tmp93 < 0
    tmp96 = tl.where(tmp95, tmp94, tmp93)
    tmp97 = tl.load(in_ptr22 + (tmp96 + 2*tmp92 + 4*x5), xmask, eviction_policy='evict_last')
    tmp99 = tmp98 + tmp89
    tmp100 = tmp98 < 0
    tmp101 = tl.where(tmp100, tmp99, tmp98)
    tmp102 = tl.load(in_ptr22 + (tmp101 + 2*tmp92 + 4*x5), xmask, eviction_policy='evict_last')
    tmp103 = tmp102 - tmp97
    tmp105 = tmp103 * tmp104
    tmp106 = tmp97 + tmp105
    tmp108 = tmp107 + tmp89
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tl.load(in_ptr22 + (tmp96 + 2*tmp110 + 4*x5), xmask, eviction_policy='evict_last')
    tmp112 = tl.load(in_ptr22 + (tmp101 + 2*tmp110 + 4*x5), xmask, eviction_policy='evict_last')
    tmp113 = tmp112 - tmp111
    tmp114 = tmp113 * tmp104
    tmp115 = tmp111 + tmp114
    tmp116 = tmp115 - tmp106
    tmp118 = tmp116 * tmp117
    tmp119 = tmp106 + tmp118
    tmp120 = tmp87 + tmp119
    tmp121 = triton_helpers.maximum(tmp18, tmp120)
    tmp122 = tmp121 <= tmp39
    tl.store(out_ptr0 + (x6), tmp19, xmask)
    tl.store(in_out_ptr0 + (x6), tmp121, xmask)
    tl.store(out_ptr3 + (x6), tmp122, xmask)
