
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (4 + x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp74 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr8 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.5
    tmp13 = tmp11 + tmp12
    tmp14 = 0.06211180124223602
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 - tmp12
    tmp17 = libdevice.floor(tmp16)
    tmp18 = tmp16 - tmp17
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = 1.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp22 + tmp21
    tmp24 = -0.75
    tmp25 = tmp23 * tmp24
    tmp26 = -3.75
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp23
    tmp29 = -6.0
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30 * tmp23
    tmp32 = -3.0
    tmp33 = tmp31 - tmp32
    tmp34 = tmp9 * tmp33
    tmp36 = tmp35 + tmp1
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tmp39 = 1.25
    tmp40 = tmp22 * tmp39
    tmp41 = 2.25
    tmp42 = tmp40 - tmp41
    tmp43 = tmp42 * tmp22
    tmp44 = tmp43 * tmp22
    tmp45 = tmp44 + tmp21
    tmp46 = tmp9 * tmp45
    tmp47 = tmp34 + tmp46
    tmp49 = tmp48 + tmp1
    tmp50 = tmp48 < 0
    tmp51 = tl.where(tmp50, tmp49, tmp48)
    tmp52 = tmp21 - tmp22
    tmp53 = tmp52 * tmp39
    tmp54 = tmp53 - tmp41
    tmp55 = tmp54 * tmp52
    tmp56 = tmp55 * tmp52
    tmp57 = tmp56 + tmp21
    tmp58 = tmp9 * tmp57
    tmp59 = tmp47 + tmp58
    tmp61 = tmp60 + tmp1
    tmp62 = tmp60 < 0
    tmp63 = tl.where(tmp62, tmp61, tmp60)
    tmp64 = 2.0
    tmp65 = tmp64 - tmp22
    tmp66 = tmp65 * tmp24
    tmp67 = tmp66 - tmp26
    tmp68 = tmp67 * tmp65
    tmp69 = tmp68 + tmp29
    tmp70 = tmp69 * tmp65
    tmp71 = tmp70 - tmp32
    tmp72 = tmp9 * tmp71
    tmp73 = tmp59 + tmp72
    tmp75 = tmp74 + tmp1
    tmp76 = tmp74 < 0
    tmp77 = tl.where(tmp76, tmp75, tmp74)
    tmp79 = tmp78 + tmp1
    tmp80 = tmp78 < 0
    tmp81 = tl.where(tmp80, tmp79, tmp78)
    tmp83 = tmp82 + tmp1
    tmp84 = tmp82 < 0
    tmp85 = tl.where(tmp84, tmp83, tmp82)
    tmp86 = x1
    tmp87 = tmp86.to(tl.float32)
    tmp88 = tmp87 + tmp12
    tmp89 = tmp88 * tmp14
    tmp90 = tmp89 - tmp12
    tmp91 = libdevice.floor(tmp90)
    tmp92 = tmp90 - tmp91
    tmp93 = triton_helpers.maximum(tmp92, tmp19)
    tmp94 = triton_helpers.minimum(tmp93, tmp21)
    tmp95 = tmp94 + tmp21
    tmp96 = tmp95 * tmp24
    tmp97 = tmp96 - tmp26
    tmp98 = tmp97 * tmp95
    tmp99 = tmp98 + tmp29
    tmp100 = tmp99 * tmp95
    tmp101 = tmp100 - tmp32
    tmp102 = tmp73 * tmp101
    tmp103 = tmp94 * tmp39
    tmp104 = tmp103 - tmp41
    tmp105 = tmp104 * tmp94
    tmp106 = tmp105 * tmp94
    tmp107 = tmp106 + tmp21
    tmp108 = tmp73 * tmp107
    tmp109 = tmp102 + tmp108
    tmp110 = tmp21 - tmp94
    tmp111 = tmp110 * tmp39
    tmp112 = tmp111 - tmp41
    tmp113 = tmp112 * tmp110
    tmp114 = tmp113 * tmp110
    tmp115 = tmp114 + tmp21
    tmp116 = tmp73 * tmp115
    tmp117 = tmp109 + tmp116
    tmp118 = tmp64 - tmp94
    tmp119 = tmp118 * tmp24
    tmp120 = tmp119 - tmp26
    tmp121 = tmp120 * tmp118
    tmp122 = tmp121 + tmp29
    tmp123 = tmp122 * tmp118
    tmp124 = tmp123 - tmp32
    tmp125 = tmp73 * tmp124
    tmp126 = tmp117 + tmp125
    tl.store(in_out_ptr0 + (x4), tmp126, xmask)
