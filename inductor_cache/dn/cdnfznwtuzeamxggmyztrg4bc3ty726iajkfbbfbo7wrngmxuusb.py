
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*i64', 'in_ptr10': '*i64', 'in_ptr11': '*fp32', 'in_ptr12': '*i64', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*i64', 'in_ptr17': '*i64', 'in_ptr18': '*fp32', 'in_ptr19': '*i64', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 19, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 256) % 60)
    x3 = xindex // 15360
    x4 = (xindex % 256)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x4 + 256*(x2) + 1024*x3), tmp4, other=0.0)
    tmp6 = tl.full([1], 0, tl.int32)
    tmp7 = triton_helpers.maximum(tmp6, tmp5)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 12, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tl.load(in_ptr1 + (x4 + 256*((-4) + x2) + 2048*x3), tmp13, other=0.0)
    tmp15 = tl.load(in_ptr2 + (x1), tmp13, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.full([XBLOCK], 8, tl.int32)
    tmp17 = tmp15 + tmp16
    tmp18 = tmp15 < 0
    tmp19 = tl.where(tmp18, tmp17, tmp15)
    tmp20 = tl.load(in_ptr3 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 + tmp16
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr4 + (tmp23 + 8*tmp19 + 64*((-4) + x2) + 512*x3), tmp13, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.full([1], 0, tl.int32)
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tl.load(in_ptr5 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp27 + tmp16
    tmp29 = tmp27 < 0
    tmp30 = tl.where(tmp29, tmp28, tmp27)
    tmp31 = tl.load(in_ptr4 + (tmp30 + 8*tmp19 + 64*((-4) + x2) + 512*x3), tmp13, eviction_policy='evict_last', other=0.0)
    tmp32 = triton_helpers.maximum(tmp25, tmp31)
    tmp33 = tmp32 - tmp26
    tmp34 = tl.load(in_ptr6 + (x0), tmp13, eviction_policy='evict_last', other=0.0)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp26 + tmp35
    tmp37 = tmp36 - tmp14
    tmp38 = tl.load(in_ptr7 + (x1), tmp13, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tmp14 + tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp13, tmp40, tmp41)
    tmp43 = tmp0 >= tmp11
    tmp44 = tl.full([1], 28, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr8 + (x4 + 256*((-12) + x2) + 4096*x3), tmp46, other=0.0)
    tmp48 = tl.load(in_ptr9 + (x1), tmp46, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full([XBLOCK], 4, tl.int32)
    tmp50 = tmp48 + tmp49
    tmp51 = tmp48 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp48)
    tmp53 = tl.load(in_ptr10 + (x0), tmp46, eviction_policy='evict_last', other=0.0)
    tmp54 = tmp53 + tmp49
    tmp55 = tmp53 < 0
    tmp56 = tl.where(tmp55, tmp54, tmp53)
    tmp57 = tl.load(in_ptr11 + (tmp56 + 4*tmp52 + 16*((-12) + x2) + 256*x3), tmp46, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.full([1], 0, tl.int32)
    tmp59 = triton_helpers.maximum(tmp58, tmp57)
    tmp60 = tl.load(in_ptr12 + (x0), tmp46, eviction_policy='evict_last', other=0.0)
    tmp61 = tmp60 + tmp49
    tmp62 = tmp60 < 0
    tmp63 = tl.where(tmp62, tmp61, tmp60)
    tmp64 = tl.load(in_ptr11 + (tmp63 + 4*tmp52 + 16*((-12) + x2) + 256*x3), tmp46, eviction_policy='evict_last', other=0.0)
    tmp65 = triton_helpers.maximum(tmp58, tmp64)
    tmp66 = tmp65 - tmp59
    tmp67 = tl.load(in_ptr13 + (x0), tmp46, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 * tmp67
    tmp69 = tmp59 + tmp68
    tmp70 = tmp69 - tmp47
    tmp71 = tl.load(in_ptr14 + (x1), tmp46, eviction_policy='evict_last', other=0.0)
    tmp72 = tmp70 * tmp71
    tmp73 = tmp47 + tmp72
    tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
    tmp75 = tl.where(tmp46, tmp73, tmp74)
    tmp76 = tmp0 >= tmp44
    tmp77 = tl.full([1], 60, tl.int64)
    tmp78 = tmp0 < tmp77
    tmp79 = tl.load(in_ptr15 + (x4 + 256*((-28) + x2) + 8192*x3), tmp76, other=0.0)
    tmp80 = tl.load(in_ptr16 + (x1), tmp76, eviction_policy='evict_last', other=0.0)
    tmp81 = tl.full([XBLOCK], 2, tl.int32)
    tmp82 = tmp80 + tmp81
    tmp83 = tmp80 < 0
    tmp84 = tl.where(tmp83, tmp82, tmp80)
    tmp85 = tl.load(in_ptr17 + (x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp86 = tmp85 + tmp81
    tmp87 = tmp85 < 0
    tmp88 = tl.where(tmp87, tmp86, tmp85)
    tmp89 = tl.load(in_ptr18 + (tmp88 + 2*tmp84 + 4*((-28) + x2) + 128*x3), tmp76, eviction_policy='evict_last', other=0.0)
    tmp90 = tl.load(in_ptr19 + (x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp90 + tmp81
    tmp92 = tmp90 < 0
    tmp93 = tl.where(tmp92, tmp91, tmp90)
    tmp94 = tl.load(in_ptr18 + (tmp93 + 2*tmp84 + 4*((-28) + x2) + 128*x3), tmp76, eviction_policy='evict_last', other=0.0)
    tmp95 = tmp94 - tmp89
    tmp96 = tl.load(in_ptr20 + (x0), tmp76, eviction_policy='evict_last', other=0.0)
    tmp97 = tmp95 * tmp96
    tmp98 = tmp89 + tmp97
    tmp99 = tmp98 - tmp79
    tmp100 = tl.load(in_ptr21 + (x1), tmp76, eviction_policy='evict_last', other=0.0)
    tmp101 = tmp99 * tmp100
    tmp102 = tmp79 + tmp101
    tmp103 = tl.full(tmp102.shape, 0.0, tmp102.dtype)
    tmp104 = tl.where(tmp76, tmp102, tmp103)
    tmp105 = tl.where(tmp46, tmp75, tmp104)
    tmp106 = tl.where(tmp13, tmp42, tmp105)
    tmp107 = tl.where(tmp4, tmp9, tmp106)
    tl.store(out_ptr0 + (x5), tmp107, None)
